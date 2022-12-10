import os

import pathlib 
import openai
import prodigy
import srsly
import spacy
from prodigy.components.preprocess import add_tokens
from dotenv import load_dotenv
from rich import print
from rich.panel import Panel

# Set up openai
load_dotenv()  # take environment variables from .env.
openai.organization = os.getenv("OPENAI_ORG")
openai.api_key = os.getenv("OPENAI_KEY")


def generate_prompt(task, examples, verbose=False):
    result = f"{task}\nI want more examples of sentences like below.\n\nCurrent examples:\n"
    for ex in examples:
        result = f"{result}\n- {ex['text']}"
    if verbose:
        print(Panel(result, title="Prompt to OpenAI"))
    return result

def make_batch(task, examples, model, verbose=True):
    prompt = generate_prompt(task=task, examples=examples, verbose=verbose)
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=0.5,
        max_tokens=100,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    resp_text = response["choices"][0]["text"]
    print(response)
    if verbose:
        print(Panel(resp_text, title="Response from OpenAI"))
    for line in resp_text.split("\n"):
        if "-" in line:
            yield {"text": line[line.find("-")+1:].strip()}


def gen_candidates(task, examples, model, verbose=False):
    while True:
        for ex in make_batch(task=task, examples=examples, model=model, verbose=verbose):
            yield ex

@prodigy.recipe(
    "spancat.openai.paraphrase",
    dataset=("Dataset to save answers to", "positional", None, str),
    filepath=("File to jsonl data to paraphrase", "positional", None, str),
    lang=("Language to use", "positional", None, str),
    taskfile=("File that describes the task.", "positional", None, str),
    labels=("Labels for spancat", "option", "l", str),
    verbose=("Show requests/responses of prompts to OpenAI", "flag", "v", bool),
    model=("GPT-3 model to use for completion", "option", "m", str),
)
def ner_openai_correct(
    dataset, filepath, lang, taskfile, labels, verbose=False, model="text-davinci-003"
):
    # Load your own streams from anywhere you want
    labels = labels.split(",")
    nlp = spacy.blank(lang)
    examples = list(srsly.read_jsonl(filepath))
    task = pathlib.Path(taskfile).read_text()
    stream = gen_candidates(task, examples, model, verbose)
    stream = add_tokens(nlp, stream, skip=True)

    def update(answers):
        for ex in answers:
            if ex['answer'] == 'accept':
                examples.append({'text': ex['text']})

    return {
        "dataset": dataset,
        "view_id": "spans_manual",
        "stream": stream,
        "update": update,
        "config": {"labels": labels, "batch_size": 1},
    }
