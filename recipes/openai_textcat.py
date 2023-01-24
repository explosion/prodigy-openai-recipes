from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import prodigy
import spacy
import srsly
import tqdm
from dotenv import load_dotenv
from jinja2 import Template
from prodigy.util import msg
from spacy.language import Language

from recipes.openai import OpenAISuggester, PromptExample, get_api_credentials
from recipes.openai import load_template, normalize_label
from recipes.openai import read_prompt_examples

CSS_FILE_PATH = Path(__file__).parent / "style.css"
DEFAULT_PROMPT_PATH = (
    Path(__file__).parent.parent / "templates" / "textcat_prompt.jinja2"
)
HTML_TEMPLATE = """
<div class="cleaned">
  {{ #label }}
    <centering>
    <h2>OpenAI ChatGPT says: {{ meta.answer }}</h2>
    </centering>
  {{ /label }}
  <details>
    <summary>Show the prompt for OpenAI</summary>
    <pre>{{openai.prompt}}</pre>
  </details>
  <details>
    <summary>Show the response from OpenAI</summary>
    <pre>{{openai.response}}</pre>
  </details>
</div>
"""

# Set up openai
load_dotenv()  # take environment variables from .env.


@dataclass
class TextCatPromptExample(PromptExample):
    """An example to be passed into an OpenAI TextCat prompt."""

    text: str
    answer: str
    reason: str

    @classmethod
    def from_prodigy(cls, example: Dict, labels: Iterable[str]) -> "PromptExample":
        """Create a prompt example from Prodigy's format."""
        if "text" not in example:
            raise ValueError("Cannot make PromptExample without text")

        full_text = example["text"]
        reason = example["meta"].get("reason")
        if len(labels) == 1:
            answer = "accept" if example.get("answer") else "reject"
        else:
            answer = example.get("accept")
        return cls(text=full_text, answer=answer, reason=reason)


def make_textcat_response_parser(labels: List[str]) -> Callable:
    def _parse_response(text: str) -> Dict:
        response: Dict[str, str] = {}
        if text and any(k in text.lower() for k in ("answer", "reason")):
            for line in text.strip().split("\n"):
                if line and ":" in line:
                    key, value = line.split(":", 1)
                    response[key.strip().lower()] = value.strip()
        else:
            response = {"answer": "", "reason": ""}

        example = _fmt_binary(response) if len(labels) == 1 else _fmt_multi(response)
        return example

    def _fmt_binary(response: Dict[str, str]) -> Dict:
        """Parse binary TextCat where the 'answer' key means it's a positive class."""
        return {
            "answer": response["answer"].lower(),
            "label": labels[0],
            "meta": {
                "answer": response["answer"].upper(),
                "reason": response["reason"],
            },
        }

    def _fmt_multi(response: Dict[str, str]) -> Dict:
        """Parse multilabel TextCat where the 'accept' key is a list of positive labels."""
        return {
            "options": [{"id": label, "text": label} for label in labels],
            "answer": "accept",
            "meta": {"reason": response.get("reason", "")},
            "accept": list(
                filter(
                    None,
                    [normalize_label(s.strip()) for s in response["answer"].split(",")],
                )
            ),
        }

    return _parse_response


@prodigy.recipe(
    # fmt: off
    "textcat.openai.correct",
    dataset=("Dataset to save answers to", "positional", None, str),
    input_path=("Path to jsonl data to annotate", "positional", None, Path),
    labels=("Labels (comma delimited)", "option", "L", lambda s: s.split(",")),
    lang=("Language to initialize spaCy model", "option", "l", str),
    model=("GPT-3 model to use for completion", "option", "m", str),
    batch_size=("Batch size to send to OpenAI API", "option", "b", int),
    segment=("Split sentences", "flag", "S", bool),
    prompt_path=("Path to the .jinja2 prompt template", "option", "p", Path),
    examples_path=("Examples file to help define the task", "option", "e", Path),
    max_examples=("Max examples to include in prompt", "option", "n", int),
    exclusive_classes=("Make the classification task exclusive", "flag", "E", bool),
    verbose=("Print extra information to terminal", "flag", "v", bool),
    # fmt: on
)
def textcat_openai_correct(
    dataset: str,
    input_path: Path,
    labels: List[str] = None,
    lang: str = "en",
    model: str = "text-davinci-003",
    batch_size: int = 10,
    segment: bool = False,
    prompt_path: Path = DEFAULT_PROMPT_PATH,
    examples_path: Optional[Path] = None,
    max_examples: int = 2,
    exclusive_classes: bool = False,
    verbose: bool = False,
):
    api_key, api_org = get_api_credentials(model)
    examples = read_prompt_examples(examples_path, example_class=TextCatPromptExample)
    if labels is None:
        msg.fail("textcat.teach requires at least one --label", exits=1)
    nlp = spacy.blank(lang)

    if segment:
        nlp.add_pipe("sentencizer")

    if not exclusive_classes and len(labels) == 1:
        msg.warn(
            "Binary classification should always be exclusive. Setting "
            "`exclusive_classes` parameter to True"
        )
        exclusive_classes = True

    # Create partial render of the template
    template = Template(
        load_template(prompt_path).render(
            exclusive_classes=exclusive_classes, labels=labels, examples=examples
        )
    )

    # Create OpenAISuggester with ChatGPT parameters
    openai = OpenAISuggester(
        response_parser=make_textcat_response_parser(labels=labels),
        prompt_template=template,
        labels=labels,
        max_examples=max_examples,
        segment=segment,
        openai_api_org=api_org,
        openai_api_key=api_key,
        openai_n=1,
        openai_model=model,
        openai_retry_timeout_s=10,
        openai_read_timeout_s=20,
        openai_n_retries=10,
        verbose=verbose,
    )
    for eg in examples:
        openai.add_example(eg)

    # Set up the stream
    stream = list(srsly.read_jsonl(input_path))
    stream = openai(tqdm.tqdm(stream), batch_size=batch_size, nlp=nlp)

    # Set up the Prodigy UI
    return {
        "dataset": dataset,
        "view_id": "blocks",
        "stream": stream,
        "update": openai.update,
        "config": {
            "labels": openai.labels,
            "batch_size": batch_size,
            "exclude_by": "input",
            "choice_style": "single" if exclusive_classes else "multiple",
            "blocks": [
                {"view_id": "classification" if len(labels) == 1 else "choice"},
                {"view_id": "html", "html_template": HTML_TEMPLATE},
            ],
            "show_flag": True,
            "global_css": CSS_FILE_PATH.read_text(),
        },
    }


@prodigy.recipe(
    # fmt: off
    "textcat.openai.fetch",
    input_path=("Path to jsonl data to annotate", "positional", None, Path),
    output_path=("Path to save the output", "positional", None, Path),
    labels=("Labels (comma delimited)", "option", "L", lambda s: s.split(",")),
    lang=("Language to use for tokenizer.", "option", "l", str),
    model=("GPT-3 model to use for completion", "option", "m", str),
    prompt_path=("Path to jinja2 prompt template", "option", "p", Path),
    examples_path=("Examples file to help define the task", "option", "e", Path),
    max_examples=("Max examples to include in prompt", "option", "n", int),
    batch_size=("Batch size to send to OpenAI API", "option", "b", int),
    segment=("Split sentences", "flag", "S", bool),
    exclusive_classes=("Make the classification task exclusive", "flag", "E", bool),
    verbose=("Print extra information to terminal", "flag", "v", bool),
    # fmt: on
)
def textcat_openai_fetch(
    input_path: Path,
    output_path: Path,
    labels: List[str] = None,
    lang: str = "en",
    model: str = "text-davinci-003",
    batch_size: int = 10,
    segment: bool = False,
    prompt_path: Path = DEFAULT_PROMPT_PATH,
    examples_path: Optional[Path] = None,
    max_examples: int = 2,
    exclusive_classes: bool = False,
    verbose: bool = False,
):
    api_key, api_org = get_api_credentials(model)
    examples = read_prompt_examples(examples_path, example_class=TextCatPromptExample)
    if labels is None:
        msg.fail("textcat.teach requires at least one --label", exits=1)
    nlp = spacy.blank(lang)

    if segment:
        nlp.add_pipe("sentencizer")

    if not exclusive_classes and len(labels) == 1:
        msg.warn(
            "Binary classification should always be exclusive. Setting "
            "`exclusive_classes` parameter to True"
        )
        exclusive_classes = True

    # Create partial render of the template
    template = Template(
        load_template(prompt_path).render(
            exclusive_classes=exclusive_classes,
            labels=labels,
            examples=examples,
        )
    )

    # Create OpenAISuggester with ChatGPT parameters
    openai = OpenAISuggester(
        response_parser=make_textcat_response_parser(labels=labels),
        prompt_template=template,
        labels=labels,
        max_examples=max_examples,
        segment=segment,
        openai_api_org=api_org,
        openai_api_key=api_key,
        openai_n=1,
        openai_model=model,
        openai_retry_timeout_s=10,
        openai_read_timeout_s=20,
        openai_n_retries=10,
        verbose=verbose,
    )
    for eg in examples:
        openai.add_example(eg)

    # Set up the stream
    stream = list(srsly.read_jsonl(input_path))
    stream = openai(tqdm.tqdm(stream), batch_size=batch_size, nlp=nlp)
    srsly.write_jsonl(output_path, stream)
