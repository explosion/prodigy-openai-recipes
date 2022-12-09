import os

import copy
import openai
import prodigy
import spacy
from spacy.language import Language
import srsly
from dotenv import load_dotenv
import rich
from rich.panel import Panel
from typing import List, Tuple, Iterable, Dict
from srsly.util import JSONOutput
from prodigy.components.preprocess import add_tokens
from prodigy.util import set_hashes

# Set up openai
load_dotenv()  # take environment variables from .env.
openai.organization = os.getenv("OPENAI_ORG")
openai.api_key = os.getenv("OPENAI_KEY")

def chunkify(nlp, stream, limit=200):
    """Split the input into chunks, where each chunk is under a certain number
    of tokens.

    Will split at sentence boundaries. If a single sentence is over the chunk
    limit, it will be emitted as one chunk.
    """
    for example in stream:
        chunk = []
        doc = nlp(example["text"])
        for sent in doc.sents:
            chunklen = sum(len(ss) for ss in chunk)
            if chunklen + len(sent) >= limit:
                yield {**example, "text": ' '.join(ss.text for ss in chunk)}
                chunk = []
            chunk.append(sent)
        yield {**example, "text": ' '.join(ss.text for ss in chunk)}

def _openai_zero_shot_ner(stream: Iterable[JSONOutput], *, model: str, labels: List[str], verbose: bool) -> Iterable[JSONOutput]:
    """Get zero-shot suggested NER annotations from OpenAI.

    Given a stream of example dictionaries, we calculate a 
    prompt, get a response from OpenAI, and add them to the
    dictionary. A further function then takes care of parsing
    the response and setting up the span annotations for Prodigy.
    """
    for example in stream:
        assert isinstance(example, dict)
        example = copy.deepcopy(example)
        prompt = generate_prompt(labels=labels, sentence=example["text"])
        if verbose:
            rich.print(Panel(prompt, title="Prompt to OpenAI"))

        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            temperature=0,
            max_tokens=64,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        resp_text = response["choices"][0]["text"]
        example["openai"] = {"prompt": prompt, "response": resp_text}
        if verbose:
            rich.print(Panel(resp_text, title="Response from OpenAI"))
        yield example


def _convert_ner_suggestions(stream: Iterable[JSONOutput], nlp: Language) -> Iterable[JSONOutput]:
    stream = add_tokens(nlp, stream, skip=True) # type: ignore
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


def generate_prompt(sentence: str, labels: List[str]) -> str:
    result = (
        "From the text below, extract the following entities in the following format:"
    )
    for label in labels:
        result = f"{result}\n{label.title()}: <comma-separated list of each {label} mentioned>"
    result = f'{result}\n\nText:\n"""\n{sentence}\n"""\n\nAnswer:\n'
    return result

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
    """Given a list of substrings, find their character start and end positions in a text. The substrings are assumed to be sorted by the order of their occurrence in the text.
    """
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


def make_candidate(example, model, nlp, labels, verbose=True):
    prompt = generate_prompt(labels=labels, sentence=example["text"], verbose=verbose)
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=0,
        max_tokens=64,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    resp_text = response["choices"][0]["text"]
    if verbose:
        print(Panel(resp_text, title="Response from OpenAI"))
    entity_lines = resp_text.split("\n")
    spans = []
    doc = nlp(example["text"])
    for line in entity_lines:
        label_name, ents = line.split(":", 1)
        processor = KeywordProcessor()
        for ent in ents.split(","):
            processor.add_keyword(ent.strip())
        for (txt, start, end) in processor.extract_keywords(
            example["text"], span_info=True
        ):
            span = doc.char_span(start, end, alignment_mode="contract")
            spans.append(
                {
                    "label": label_name,
                    "start": start,
                    "end": end,
                    "token_start": span.start,
                    "token_end": span.end - 1,
                }
            )
    return {**example, "spans": spans}


@prodigy.recipe(
    "ner.openai.correct",
    dataset=("Dataset to save answers to", "positional", None, str),
    filepath=("File to jsonl data to annotate", "positional", None, str),
    lang=("Language to use for tokenizer.", "positional", None, str),
    labels=("Labels ", "option", "l", str),
    verbose=("Show requests/responses of prompts to OpenAI", "flag", "v", bool),
    model=("GPT-3 model to use for completion", "option", "m", str),
)
def ner_openai_correct(
    dataset, filepath, lang, labels, verbose=False, model="text-davinci-002"
):
    # Load your own streams from anywhere you want
    labels = labels.split(",")
    nlp = spacy.blank(lang)
    nlp.add_pipe("sentencizer")
    stream = srsly.read_jsonl(filepath)
    stream = chunkify(nlp, stream)
    stream = _openai_zero_shot_ner(stream, model=model, labels=labels, verbose=verbose)
    stream = _convert_ner_suggestions(stream, nlp=nlp)

    return {
        "dataset": dataset,
        "view_id": "ner_manual",
        "stream": stream,
        "config": {"labels": labels, "batch_size": 1},
    }
