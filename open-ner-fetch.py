import copy
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import httpx
import spacy
import srsly
from dotenv import load_dotenv
import prodigy
from prodigy.components.preprocess import add_tokens
from prodigy.util import set_hashes
from spacy.language import Language
from srsly.util import JSONOutput
import tqdm
import time

# Set up openai
load_dotenv()  # take environment variables from .env.


@prodigy.recipe(
    "ner.openai.fetch",
    filepath_in=("File to jsonl input data", "positional", None, Path),
    filepath_out=("File to jsonl output data", "positional", None, Path),
    labels=("Labels, seperated by a comma ", "positional", None, str),
    model=("GPT-3 model to use for completion", "option", "m", str),
    batch_size=("Batch size to send to OpenAI", "option", "b", int),
)
def fetch_ner(
    filepath_in: Path,
    filepath_out: Path,
    labels: str,
    model: str = "text-davinci-003",
    batch_size: int = 10,
):
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_KEY')}",
        "OpenAI-Organization": os.getenv("OPENAI_ORG"),
        "Content-Type": "application/json",
    }

    stream = list(srsly.read_jsonl(filepath_in))
    labels_list = labels.split(",")
    for batch in tqdm.tqdm(_batch_sequence(stream, batch_size)):
        prompts = [_get_prompt(eg["text"], labels_list) for eg in batch]
        r = httpx.post(
            "https://api.openai.com/v1/completions",
            headers=headers,
            json={
                "model": model,
                "prompt": prompts,
                "temperature": 0,
                "max_tokens": 500,
            },
        )
        r.raise_for_status()
        responses = r.json()
        for i, eg in enumerate(batch):
            prompt = prompts[i]
            response = responses["choices"][i]["text"]
            eg["openai"] = {"prompt": prompt, "response": response}

        time.sleep(1)
    srsly.write_jsonl(filepath_out, stream)


def _batch_sequence(items: List, batch_size: int) -> List[List]:
    output = []
    i = 0
    while i < len(items):
        output.append(items[i : i + batch_size])
        i += len(output[-1])
    return output


def _get_prompt(text: str, labels: List[str]) -> str:
    """Compute the zero-shot NER prompt for text"""
    result = (
        "From the text below, extract the following entities in the following format:"
    )
    for label in labels:
        result = f"{result}\n{label.title()}: <comma-separated list of each {label} mentioned>"
    result = f'{result}\n\nText:\n"""\n{text}\n"""\n\nAnswer:\n'
    return result


@prodigy.recipe(
    "ner.openai.convert",
    filepath_in=("File to jsonl input data", "positional", None, str),
    filepath_out=("File to jsonl output data", "positional", None, str),
    lang=("Language to use for tokenizer.", "positional", None, str),
    labels=("Labels, seperated by a comma ", "option", "l", str),
)
def convert_ner_suggestions(filepath_in, filepath_out, lang, labels):
    stream = list(srsly.read_jsonl(filepath_in))
    labels = labels.split(",")
    nlp = spacy.blank(lang)
    stream = _convert_openai_ner_suggestions(stream, nlp)
    srsly.write_jsonl(filepath_out, stream)


def _convert_openai_ner_suggestions(
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
    text = text.lower()
    offsets = []
    search_from = 0
    for substring in substrings:
        substring = substring.lower().strip()
        # Find from an offset, to handle phrases that
        # occur multiple times in the text.
        start = text.find(substring, search_from)
        if start != -1:
            end = start + len(substring)
            offsets.append((start, end))
            search_from = end
    return offsets
