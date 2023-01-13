import os
import time
from pathlib import Path
from typing import Callable, List, TypeVar

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

_ItemT = TypeVar("_ItemT")

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


def _parse_terms(completion):
    # Sometimes it only returns a single item. For example, when there are
    # many, many seeds around.
    if "\n" not in completion:
        return [completion.replace("-", "").strip().lower()]
    # Other times we cannot assume the final item will have had sufficient 
    # tokens available to complete the term, so we have to discard it.
    lines = completion.split("\n")
    return [item.replace("-", "").strip().lower() for item in lines][:-1]


@prodigy.recipe(
    "terms.openai.fetch",
    query=("Query to send to OpenAI", "positional", None, str),
    output_path=("Path to save the output", "positional", None, Path),
    seeds=("One of more comma-seperated seed phrases.","option","s",lambda d: d.split(","),),
    n=("Number of items to generate", "option", "n", int),
    model=("GPT-3 model to use for completion", "option", "m", str),
    prompt_path=("Path to jinja2 prompt template", "option", "p", Path),
    verbose=("Print extra information to terminal", "flag", "v", bool),
    resume=("Resume by loading in text examples from output file.", "flag", "r", bool),
    progress=("Print progress of the recipe.", "flag", "pb", bool),
    max_tokens=("Max tokens to generate", "option", "t", int),
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
    max_tokens=100,
):
    """Get bulk term suggestions from an OpenAI API, using zero-shot learning.
    The results can then be corrected using the `terms.openai.correct` recipe.

    This approach lets you get the openai queries out of the way upfront, which can help
    if you want to use multiple annotators of if you want to make sure you don't have to
    wait on the OpenAI queries. The downside is that you can't flag examples to be integrated
    into the prompt during the annotation.
    """
    template = _load_template(prompt_path)
    terms = []
    if resume:
        examples = srsly.read_jsonl(output_path)
        seeds.extend([e["text"] for e in examples])
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_KEY')}",
        "OpenAI-Organization": os.getenv("OPENAI_ORG"),
        "Content-Type": "application/json",
    }
    while len(terms) < n:
        prompt = template.render(n=n, examples=seeds + terms, description=query)
        if verbose:
            rich.print(Panel(prompt, title="Prompt to OpenAI"))
        resp = httpx.post(
            "https://api.openai.com/v1/completions",
            headers=headers,
            json={
                "model": model,
                "prompt": [prompt],
                "temperature": 1,
                "max_tokens": max_tokens,
            },
            timeout=30,
        )
        completion = resp.json()["choices"][0]["text"]
        parsed_terms = _parse_terms(completion=completion)
        srsly.write_jsonl(
            output_path,
            [{"text": t, "meta": {"openai_query": query}} for t in parsed_terms],
            append=True,
            append_new_line=False,
        )
        terms.extend(parsed_terms)
        if verbose:
            rich.print(Panel(Pretty(terms), title="Terms collected sofar."))
        if progress:
            rich.print(f"Received {len(parsed_terms)} items, totalling {len(terms)} terms. Progress at {round(len(terms)/n*100)}%.")
