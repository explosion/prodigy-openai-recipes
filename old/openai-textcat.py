import os

import openai
import prodigy
import srsly
from dotenv import load_dotenv
from rich import print
from rich.panel import Panel

# Set up openai
load_dotenv()  # take environment variables from .env.
openai.organization = os.getenv("OPENAI_ORG")
openai.api_key = os.getenv("OPENAI_KEY")


def generate_prompt(labels, sentence, verbose=False):
    result = (
        "From the text below, tell me which class describes it best. From the following list:"
    )
    for label in labels:
        result = f"{result}\n- {label.lower()}"
    result = f'{result}\n\nText:\n"""\n{sentence}\n"""\n\nAnswer:\n'
    if verbose:
        print(Panel(result, title="Prompt to OpenAI"))
    return result


def make_candidate(example, model, labels, verbose=True):
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
    return {**example, "label": resp_text.strip()}


@prodigy.recipe(
    "textcat.openai.correct",
    dataset=("Dataset to save answers to", "positional", None, str),
    filepath=("File to jsonl data to annotate", "positional", None, str),
    labels=("Labels ", "option", "l", str),
    verbose=("Show requests/responses of prompts to OpenAI", "flag", "v", bool),
    model=("GPT-3 model to use for completion", "option", "m", str),
)
def ner_openai_correct(
    dataset, filepath, lang, labels, verbose=False, model="text-davinci-002"
):
    # Load your own streams from anywhere you want
    labels = labels.split(",")
    stream = srsly.read_jsonl(filepath)
    stream = (
        make_candidate(s, model=model, labels=labels, verbose=verbose)
        for s in stream
    )

    return {
        "dataset": dataset,
        "view_id": "classification",
        "stream": stream,
        "config": {"labels": labels, "batch_size": 1},
    }
