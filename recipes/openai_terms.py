import os
import time
from functools import reduce
from pathlib import Path
from typing import Callable, Dict, List

import httpx
import jinja2
import prodigy
import prodigy.components.db
import prodigy.components.preprocess
import prodigy.util
import rich
import srsly
from dotenv import load_dotenv
from prodigy.util import msg
from rich.panel import Panel
from rich.pretty import Pretty

DEFAULT_PROMPT_PATH = Path(__file__).parent.parent / "templates" / "terms_prompt.jinja2"

# Set up openai
load_dotenv()  # take environment variables from .env.


def _load_template(path: Path) -> jinja2.Template:
    # I know jinja has a lot of complex file loading stuff,
    # but we're not using the inheritance etc that makes
    # that stuff worthwhile.
    if not path.suffix == ".jinja2":
        msg.fail(
            "The --prompt-path (-p) parameter expects a .jinja2 file.",
            exits=1,
        )
    with path.open("r", encoding="utf8") as file_:
        text = file_.read()
    return jinja2.Template(text)


def _parse_terms(completion: str) -> List[str]:
    if "\n" not in completion:
        # Sometimes it only returns a single item. For example, when there are
        # many, many seeds around.
        lines = [completion]
    else:
        # Other times we cannot assume the final item will have had sufficient
        # tokens available to complete the term, so we have to discard it.
        lines = [item for item in completion.split("\n") if len(item)]
        lines = lines[:-1]
    return [item.replace("-", "").strip() for item in lines]


def _retry429(
    call_api: Callable[[], httpx.Response], n: int, timeout_s: int
) -> httpx.Response:
    """Retry a call to the OpenAI API if we get a 429: Too many requests
    error.
    """
    assert n >= 0
    assert timeout_s >= 1
    r = call_api()
    i = -1
    while i < n and r.status_code == 429:
        time.sleep(timeout_s)
        i += 1
    return r


@prodigy.recipe(
    # fmt: off
    "terms.openai.fetch",
    query=("Query to send to OpenAI", "positional", None, str),
    output_path=("Path to save the output", "positional", None, Path),
    seeds=("One or more comma-seperated seed phrases.","option","s",lambda d: d.split(",")),
    n=("Number of items to generate", "option", "n", int),
    model=("GPT-3 model to use for completion", "option", "m", str),
    prompt_path=("Path to jinja2 prompt template", "option", "p", Path),
    verbose=("Print extra information to terminal", "flag", "v", bool),
    resume=("Resume by loading in text examples from output file.", "flag", "r", bool),
    progress=("Print progress of the recipe.", "flag", "pb", bool),
    temperature=("OpenAI temperature param", "option", "t", float),
    top_p=("OpenAI top_p param", "option", "tp", float),
    best_of=("OpenAI best_of param", "option", "bo", int),
    n_batch=("OpenAI batch size param", "option", "nb", int),
    max_tokens=("Max tokens to generate per call", "option", "mt", int),
    # fmt: on
)
def terms_openai_fetch(
    query: str,
    output_path: Path,
    seeds: List[str] = [],
    n: int = 100,
    model: str = "text-davinci-003",
    prompt_path: Path = DEFAULT_PROMPT_PATH,
    verbose: bool = False,
    resume: bool = False,
    progress: bool = False,
    temperature=1.0,
    top_p=1.0,
    best_of=10,
    n_batch=10,
    max_tokens=100,
):
    """Get bulk term suggestions from an OpenAI API, using zero-shot learning.
    The results can then be corrected using the `terms.openai.correct` recipe.

    This approach lets you get the openai queries out of the way upfront, which can help
    if you want to use multiple annotators of if you want to make sure you don't have to
    wait on the OpenAI queries. The downside is that you can't flag examples to be integrated
    into the prompt during the annotation.
    """
    tic = time.time()
    template = _load_template(prompt_path)
    # The `best_of` param cannot be less than the amount we batch.
    if best_of < n_batch:
        best_of = n_batch

    # Start collection of terms. If we resume we also fill seed terms with file contents.
    terms = []
    if resume:
        if output_path.exists():
            examples = srsly.read_jsonl(output_path)
            terms.extend([e["text"] for e in examples])

    # Mimic behavior from Prodigy terms recipe to ensure that seed terms also appear in output
    for seed in seeds:
        if seed not in terms:
            srsly.write_jsonl(
                output_path,
                [{"text": seed, "meta": {"openai_query": query}}],
                append=True,
                append_new_line=False,
            )

    # Ensure we have access to correct environment variables and construct headers
    if not os.getenv("OPENAI_KEY"):
        msg.fail("The `OPENAI_KEY` is missing from your `.env` file.", exits=1)

    if not os.getenv("OPENAI_ORG"):
        msg.fail("The `OPENAI_ORG` is missing from your `.env` file.", exits=1)

    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_KEY')}",
        "OpenAI-Organization": os.getenv("OPENAI_ORG"),
        "Content-Type": "application/json",
    }

    # This recipe may overshoot the target, but we keep going until we have at least `n`
    while len(terms) < n:
        prompt = template.render(n=n, examples=seeds + terms, description=query)
        if verbose:
            rich.print(Panel(prompt, title="Prompt to OpenAI"))

        make_request = lambda: httpx.post(
            "https://api.openai.com/v1/completions",
            headers=headers,
            json={
                "model": model,
                "prompt": [prompt],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "n": min(n_batch, best_of),
                "best_of": best_of,
            },
            timeout=30,
        )

        # Catch 429: too many request errors
        resp = _retry429(make_request, n=1, timeout_s=30)

        # Report on any other error that might happen, the most typical use-case is
        # catching the maximum context length of 4097 tokens when the prompt gets big.
        if resp.status_code != 200:
            msg.fail(f"Received status code {resp.status_code} from OpenAI. Details:")
            rich.print(resp.json())
            exit(code=1)

        # Cast to a set to make sure we remove duplicates
        choices = resp.json()["choices"]
        sets_of_terms = [set(_parse_terms(c["text"])) for c in choices]
        parsed_terms = list(reduce(lambda a, b: a.union(b), sets_of_terms))

        # Save intermediate results into file, in-case of a hiccup
        srsly.write_jsonl(
            output_path,
            [{"text": t, "meta": {"openai_query": query}} for t in parsed_terms],
            append=True,
            append_new_line=False,
        )

        # Make the terms list bigger and re-use terms in next prompt.
        terms.extend(parsed_terms)
        if verbose:
            rich.print(Panel(Pretty(terms), title="Terms collected sofar."))
        if progress:
            rich.print(
                f"Received {len(parsed_terms)} items, totalling {len(terms)} terms. "
                f"Progress at {round(len(terms)/n*100)}% after {round(time.time() - tic)}s."
            )
