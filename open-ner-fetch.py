import asyncio
import copy
import itertools
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import httpx
import openai
import spacy
import srsly
from dotenv import load_dotenv
import prodigy
from prodigy.components.preprocess import add_tokens
from prodigy.util import set_hashes
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from spacy.language import Language
from srsly.util import JSONOutput

# Set up openai
load_dotenv()  # take environment variables from .env.
openai.organization = os.getenv("OPENAI_ORG")
openai.api_key = os.getenv("OPENAI_KEY")

headers = {
    "Authorization": f"Bearer {os.getenv('OPENAI_KEY')}",
    "OpenAI-Organization": os.getenv("OPENAI_ORG"),
    "Content-Type": "application/json",
}


async def fetch_completion(example: Dict[str, str]):
    """Send a single completion request based on an example"""
    json_data = {
        "model": example["openai"]["model"],
        "prompt": example["openai"]["prompt"],
        "temperature": 0,
        "max_tokens": 100,
    }
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://api.openai.com/v1/completions", headers=headers, json=json_data
        )
        return r


async def request_many(stream):
    """Select 10 items from the stream and batch in a single async call"""
    top10 = list(itertools.islice(stream, 10))
    responses = await asyncio.gather(*[fetch_completion(ex) for ex in top10])
    for ex, resp in zip(top10, responses):
        ex["openai"]["response"] = resp.json()["choices"][0]["text"]
    return top10


def attach_prompt(example: Dict[str, str], labels: List[str], model: str) -> str:
    """Attach the openai prompt to each example"""
    result = (
        "From the text below, extract the following entities in the following format:"
    )
    for label in labels:
        result = f"{result}\n{label.title()}: <comma-separated list of each {label} mentioned>"
    result = f'{result}\n\nText:\n"""\n{example["text"]}\n"""\n\nAnswer:\n'
    example["openai"] = {"prompt": result, "model": model}
    return example


def _convert_ner_suggestions(
    stream: Iterable[JSONOutput], nlp: Language
) -> Iterable[JSONOutput]:
    stream = add_tokens(nlp, stream, skip=True)  # type: ignore
    for example in stream:
        example = copy.deepcopy(example)
        doc = nlp.make_doc(example["text"])
        response = _parse_response(example["openai"]["response"])
        spans = []
        for label, phrases in response:
            offsets = _find_substrings(doc.text, phrases)
            for start, end in offsets:
                span = doc.char_span(start, end, alignment_mode="contract")
                if span is not None:
                    spans.append(
                        {
                            "label": label,
                            "start": start,
                            "end": end,
                            "token_start": span.start,
                            "token_end": span.end - 1,
                        }
                    )
        example = set_hashes({**example, "spans": spans})
        yield example


def _parse_response(text: str) -> List[Tuple[str, List[str]]]:
    """Interpret OpenAI's NER response. It's supposed to be
    a list of lines, with each line having the form:
    Label: phrase1, phrase2, ...

    However, there's no guarantee that the model will give
    us well-formed output. It could say anything, it's an LM.
    So we need to be robust.
    """
    output = []
    for line in text.strip().split("\n"):
        if line and ":" in line:
            label, phrases = line.split(":", 1)
            if phrases.strip():
                output.append((label, phrases.strip().split(",")))
    return output


def _find_substrings(text: str, substrings: List[str]) -> List[Tuple[int, int]]:
    """Given a list of substrings, find their character start and end positions in a text. The substrings are assumed to be sorted by the order of their occurrence in the text."""
    # Sometimes the first substring is title-cased.
    # If it is, and it's not in the text titled, fix it.
    if substrings and substrings[0] not in text:
        substrings[0] = substrings[0].lower()
    offsets = []
    search_from = 0
    for substring in substrings:
        # Find from an offset, to handle phrases that
        # occur multiple times in the text.
        start = text.find(substring, search_from)
        if start != -1:
            end = start + len(substring)
            offsets.append((start, end))
            search_from = end
    return offsets


@prodigy.recipe(
    "ner.openai.fetch",
    filepath_in=("File to jsonl input data", "positional", None, str),
    filepath_out=("File to jsonl output data", "positional", None, str),
    lang=("Language to use for tokenizer.", "positional", None, str),
    labels=("Labels, seperated by a comma ", "option", "l", str),
    n_examples=("Number of examples to fetch", "option", "n", int),
    verbose=("Show requests/responses of prompts to OpenAI", "flag", "v", bool),
    model=("GPT-3 model to use for completion", "option", "m", str),
)
def main(filepath_in, filepath_out, lang, labels, n_examples=200, verbose=False, model="text-davinci-003"):
    stream = srsly.read_jsonl(filepath_in)
    labels = labels.split(",")
    nlp = spacy.blank(lang)
    stream = ({**ex, "labels": labels} for ex in stream)
    stream = (
        set_hashes(ex, input_keys=("text",), task_keys=("labels",)) for ex in stream
    )
    stream = (attach_prompt(ex, labels=labels, model=model) for ex in stream)

    # Make sure we don't send the same prompt twice
    # TODO: should the model used by OpenAI be part of the _task_hash? 
    annot_stream = []
    if Path(filepath_out).exists():
        annot_stream = srsly.read_jsonl(filepath_out)
    hashes_already_done = {(e["_input_hash"], e["_task_hash"]) for e in annot_stream}
    stream = (
        ex
        for ex in stream
        if (ex["_input_hash"], ex["_task_hash"]) not in hashes_already_done
    )

    examples = []

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        TextColumn("-"),
        TimeRemainingColumn(),
    )
    with progress:
        for _ in progress.track(
            range(n_examples//10), description="Sending prompts to OpenAI."
        ):
            try:
                responses = asyncio.run(request_many(stream=stream))
            except httpx.HTTPError as exc:
                progress.console.print("Hit an HTTPError. Will skip a batch.")
                if verbose:
                    progress.console.log(exc)
            for r in _convert_ner_suggestions(responses, nlp=nlp):
                examples.append(r)

    srsly.write_jsonl(filepath_out, examples, append=True, append_new_line=False)


if __name__ == "__main__":
    main("fashion_brands_training.jsonl", "prompts.jsonl", 15)
