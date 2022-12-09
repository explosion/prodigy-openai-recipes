import os
import random
import copy
import openai
import prodigy
from prodigy.components.db import connect
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


def _openai_zero_shot_ner(stream: Iterable[JSONOutput], *, model: str, labels: List[str], verbose: bool, steer_examples: List[Dict]) -> Iterable[JSONOutput]:
    """Get zero-shot suggested NER annotations from OpenAI.

    Given a stream of example dictionaries, we calculate a 
    prompt, get a response from OpenAI, and add them to the
    dictionary. A further function then takes care of parsing
    the response and setting up the span annotations for Prodigy.
    """
    for example in stream:
        assert isinstance(example, dict)
        example = copy.deepcopy(example)
        prompt = generate_prompt(labels=labels, sentence=example["text"], steer_examples=steer_examples)
        if verbose:
            rich.print(Panel(prompt, title="Prompt to OpenAI"))

        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            temperature=0.7,
            max_tokens=256,
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
        example["html"] = f'<div class="cleaned"><details><summary><b>Show the prompt for OpenAI</b></summary><p>{example["openai"]["prompt"]}</p></details><details><summary><b>Show the response from OpenAI</b></summary><p>{example["openai"]["response"]}</p></details></div>'
        yield example


def list_examples_of_type(example, label):
    return ",".join([example['text'][s['start']:s['end']] for s in example['spans'] if s['label'].lower() == label.lower()])

def generate_prompt(sentence: str, labels: List[str], steer_examples: List[Dict]) -> str:
    result = (
        "From the text below, extract the following entities in the following format:"
    )
    for label in labels:
        result = f"{result}\n{label.title()}: <comma-separated list of each {label} mentioned>"
    if steer_examples:
        example = random.choice(steer_examples)
        result = f'{result}\n\nText:\n"""\n{example["text"]}\n"""\n\nAnswer:'
        for label in labels:
            result = f"{result}\n{label.title()}: {list_examples_of_type(example, label)}"
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


def before_db(examples):
    for ex in examples:
        del ex["html"]
    return examples

@prodigy.recipe(
    "ner.openai.correct",
    dataset=("Dataset to save answers to", "positional", None, str),
    filepath=("File to jsonl data to annotate", "positional", None, str),
    lang=("Language to use for tokenizer.", "positional", None, str),
    labels=("Labels ", "option", "l", str),
    verbose=("Show requests/responses of prompts to OpenAI", "flag", "v", bool),
    model=("GPT-3 model to use for completion", "option", "m", str),
    steer=("Use flagged examples from dataset to steer output", "flag", "steer", bool)
)
def ner_openai_correct(
    dataset, filepath, lang, labels, verbose=False, model="text-davinci-003", steer=False,
):
    # Load your own streams from anywhere you want
    labels = labels.split(",")
    nlp = spacy.blank(lang)
    steer_examples = []
    if steer:
        db = connect()
        steer_examples = [ex for ex in db.get_dataset(dataset) if ex.get("flagged", False)]
    stream = srsly.read_jsonl(filepath)
    stream = _openai_zero_shot_ner(stream, model=model, labels=labels, verbose=verbose, steer_examples=steer_examples)
    stream = _convert_ner_suggestions(stream, nlp=nlp)
    return {
        "dataset": dataset,
        "view_id": "blocks",
        "stream": stream,
        "before_db": before_db,
        "config": {
            "labels": labels, 
            "batch_size": 1,
            "blocks":[
                {"view_id": "ner_manual"},
                {"view_id": "html"}
            ],
            "show_flag": True,
            "global_css": """
            .cleaned{
                text-align: left;
                font-size: 14px;
            }
            .cleaned p{
                background-color: #eeeeee;
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
                padding: 15px 20px;
                border-radius: 15px;
            }
            """
        },
    }
