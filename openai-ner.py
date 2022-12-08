import os

import openai
import prodigy
import spacy
import srsly
from dotenv import load_dotenv
from flashtext import KeywordProcessor
from prodigy.components.preprocess import add_tokens
from rich import print
from rich.panel import Panel

# Set up openai
load_dotenv()  # take environment variables from .env.
openai.organization = os.getenv("OPENAI_ORG")
openai.api_key = os.getenv("OPENAI_KEY")


def generate_prompt(labels, sentence, verbose=False):
    result = (
        "From the text below, extract the following entities in the following format:"
    )
    for label in labels:
        result = f"{result}\n{label.title()}: <comma-separated list of each {label} mentioned>"
    result = f'{result}\n\nText:\n"""\n{sentence}\n"""\n\nAnswer:\n'
    if verbose:
        print(Panel(result, title="Prompt to OpenAI"))
    return result


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
    stream = srsly.read_jsonl(filepath)
    stream = add_tokens(nlp, stream, skip=True)
    stream = (
        make_candidate(s, model=model, nlp=nlp, labels=labels, verbose=verbose)
        for s in stream
    )

    return {
        "dataset": dataset,
        "view_id": "ner_manual",
        "stream": stream,
        "config": {"labels": labels, "batch_size": 1},
    }
