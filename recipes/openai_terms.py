import os
import time
from functools import reduce
from pathlib import Path
from typing import Callable, Dict, Iterable, List, TypeVar, Union

import httpx
import jinja2
import prodigy
import prodigy.components.preprocess
import prodigy.util
import rich
import srsly
from dotenv import load_dotenv
from prodigy import set_hashes
from prodigy.components.db import connect
from prodigy.util import msg
from rich.panel import Panel
from rich.pretty import Pretty

DEFAULT_PROMPT_PATH = (
    Path(__file__).parent.parent / "templates" / "variants_prompt.jinja2"
)

# Set up openai
load_dotenv()  # take environment variables from .env.

_ItemT = TypeVar("_ItemT")


def _batch_sequence(items: Iterable[_ItemT], batch_size: int) -> Iterable[List[_ItemT]]:
    batch = []
    for eg in items:
        batch.append(eg)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


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


def _handle_numbered_list(line: str):
    if line[0].isnumeric:
        if all(char.isnumeric for char in line[: line.find(".")]):
            return line[line.find(".") + 1 :]
    return line


def _parse_variants(completion: str) -> List[str]:
    # The variants parsing is different because we have to worry about
    # token limits for the final term in the list.
    if "\n" not in completion:
        # Sometimes it only returns a single item.
        lines = [completion]
    else:
        lines = [
            _handle_numbered_list(item) for item in completion.split("\n") if len(item)
        ]
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


def _generate_headers() -> Dict[str, str]:
    if not os.getenv("OPENAI_KEY"):
        msg.fail("The `OPENAI_KEY` is missing from your `.env` file.", exits=1)

    if not os.getenv("OPENAI_ORG"):
        msg.fail("The `OPENAI_ORG` is missing from your `.env` file.", exits=1)

    return {
        "Authorization": f"Bearer {os.getenv('OPENAI_KEY')}",
        "OpenAI-Organization": os.getenv("OPENAI_ORG"),
        "Content-Type": "application/json",
    }


@prodigy.recipe(
    # fmt: off
    "terms.openai.fetch",
    query=("Query to send to OpenAI", "positional", None, str),
    output_path=("Path to save the output", "positional", None, Path),
    seeds=("One or more comma-seperated seed phrases.","option","s",lambda d: d.split(",")),
    n=("The minimum number of items to generate", "option", "n", int),
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
    """Get bulk term suggestions from the OpenAI API, using zero-shot learning.

    The results can then be corrected using the `prodigy textcat.manual` recipe and
    turned into patterns via `prodigy terms.to-patterns`.
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

    headers = _generate_headers()

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
            timeout=45,
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


@prodigy.recipe(
    # fmt: off
    "terms.openai.variants",
    query=("Query to send to OpenAI", "positional", None, str),
    input_path=("Path to save the output", "positional", None, Path),
    output_path=("Path to save the output", "positional", None, Path),
    model=("GPT-3 model to use for completion", "option", "m", str),
    prompt_path=("Path to jinja2 prompt template", "option", "p", Path),
    verbose=("Print extra information to terminal", "flag", "v", bool),
    progress=("Print progress of the recipe.", "flag", "pb", bool),
    temperature=("OpenAI temperature param", "option", "t", float),
    top_p=("OpenAI top_p param", "option", "tp", float),
    frequency_penalty=("OpenAI frequency penalty param", "option", "nb", int),
    max_tokens=("Max tokens to generate per call", "option", "mt", int),
    resume=("Don't generate examples from parents that already have children.", "flag", "r", bool),
    # fmt: on
)
def variants_openai_fetch(
    query: str,
    input_path: Union[Path, str],
    output_path: Path,
    model: str = "text-davinci-003",
    prompt_path: Path = DEFAULT_PROMPT_PATH,
    verbose: bool = False,
    progress: bool = False,
    temperature=0.1,
    top_p=1.0,
    frequency_penalty=0.85,
    max_tokens=100,
    resume=False,
):
    """Get variations on term suggestions from the OpenAI API, using zero-shot learning.

    The results can then be corrected using the `prodigy textcat.manual` recipe and
    turned into patterns via `prodigy terms.to-patterns`.
    """
    tic = time.time()
    template = _load_template(prompt_path)

    # Collect stream of parents, which could be a Prodigy dataset too
    if input_path.exists():
        parent_stream = srsly.read_jsonl(input_path)
    else:
        db = connect()
        # At this point we know it's not a file on disk, so cast to string
        input_path = str(input_path)
        if input_path not in db.datasets:
            msg.fail(
                f"The value of input_path `{input_path}` does not exist on disk and does not exist as a dataset in Prodigy. Might be a typo?",
                exits=True,
            )
        parent_stream = db.get_dataset(input_path)
        parent_stream = (ex for ex in parent_stream if ex["answer"] == "accept")

    # Collect all the parent terms to generate variations for
    # making sure that when we --resume we don't generate for the same parents.
    parent_examples = (set_hashes(e, input_keys=("text",)) for e in parent_stream)
    existing_parent_hashes = {}
    if resume:

        def add_parent_hash(stream: Iterable[_ItemT]) -> Iterable[_ItemT]:
            for ex in stream:
                parent_ex = {"text": ex["meta"]["parent"]}
                ex = set_hashes(parent_ex, input_keys=("text",))
                yield ex

        stream = srsly.read_jsonl(output_path) if output_path.exists() else []
        stream = add_parent_hash(stream)
        existing_parent_hashes = set(ex["_input_hash"] for ex in stream)

    parent_examples = (
        ex for ex in parent_examples if ex["_input_hash"] not in existing_parent_hashes
    )

    # Ensure we have access to correct environment variables and construct headers
    headers = _generate_headers()

    # This recipe may overshoot the target, but we keep going until we have at least `n`
    batched_parents = list(_batch_sequence(parent_examples, 5))
    for i, batch in enumerate(batched_parents):
        prompts = [
            template.render(example=ex["text"], description=query) for ex in batch
        ]
        if verbose:
            rich.print(Panel(prompts[0], title="Prompt to OpenAI"))

        make_request = lambda: httpx.post(
            "https://api.openai.com/v1/completions",
            headers=headers,
            json={
                "model": model,
                "prompt": prompts,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
            },
            timeout=45,
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
        examples = []
        choices = resp.json()["choices"]
        for choice, parent in zip(choices, batch):
            for term in set(_parse_variants(choice["text"])):
                example = {
                    "text": term,
                    "meta": {"openai_query": query, "parent": parent["text"]},
                }
                examples.append(example)

        # Save intermediate results into file, in-case of a hiccup
        srsly.write_jsonl(
            output_path,
            examples,
            append=True,
            append_new_line=False,
        )

        if progress:
            rich.print(
                f"Batch {i + 1} complete, totalling {len(examples)} variants. "
                f"Progress at {round(i/len(batched_parents)*100)}% after {round(time.time() - tic)}s."
            )
