from pathlib import Path
from typing import Dict, List, Optional, Iterable
from dataclasses import dataclass

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


class TextCatOpenAISuggester(OpenAISuggester):
    def parse_response(self, example: Dict, response: str, nlp: Language) -> Dict:

        # Add meta to display OpenAI 'reason'
        if "meta" not in example:
            example["meta"] = {}

        example = (
            self._parse_binary(example, response)
            if len(self.labels) == 1
            else self._parse_multi(example, response)
        )
        return example

    def _parse_output(self, text: str) -> Dict[str, str]:
        """Parse the actual text response from OpenAI."""
        output = {}
        for line in text.strip().split("\n"):
            if line and ":" in line:
                key, value = line.split(":", 1)
                output[key.strip().lower()] = value.strip()
        return output

    def _parse_binary(self, example: Dict, response: str) -> Dict:
        """Parse binary TextCat where the 'answer' key means it's a positive class."""
        output = self._parse_output(response)
        example["answer"] = output["answer"].lower() == "accept"
        example["meta"] = {
            "answer": output["answer"].upper(),
            "reason": output["reason"],
        }
        example["label"] = self.labels[0]
        return example

    def _parse_multi(self, example: Dict, response: str) -> Dict:
        """Parse multilabel TextCat."""
        output = self._parse_output(response)
        example["options"] = [{"id": label, "text": label} for label in self.labels]
        example["meta"]["reason"] = output["reason"]
        example["answer"] = "accept"
        # Filter removes any empty strings
        example["accept"] = list(
            filter(
                None, [normalize_label(s.strip()) for s in output["answer"].split(",")]
            )
        )
        return example


@prodigy.recipe(
    # fmt: off
    "textcat.openai.correct",
    input_path=("Path to jsonl data to annotate", "positional", None, Path),
    prompt_path=("Path to jinja2 prompt template", "positional", None, Path),
    labels=("Labels (comma delimited)", "option", "L", lambda s: s.split(",")),
    lang=("Language to initialize spaCy model", "option", "l", str),
    model=("GPT-3 model to use for completion", "option", "m", str),
    batch_size=("Batch size to send to OpenAI API", "option", "b", int),
    segment=("Split sentences", "flag", "S", bool),
    examples_path=("Examples file to help define the task", "option", "e", Path),
    max_examples=("Max examples to include in prompt", "option", "n", int),
    exclusive_classes=("Make the classification task exclusive", "flag", "E", bool),
    verbose=("Print extra information to terminal", "flag", "v", bool),
    # fmt: on
)
def textcat_openai_correct(
    dataset: str,
    input_path: Path,
    prompt_path: Path,
    labels: List[str] = None,
    lang: str = "en",
    model: str = "text-davinci-003",
    batch_size: int = 10,
    segment: bool = False,
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

    # Create partial render of the template
    if not exclusive_classes and len(labels) == 1:
        msg.warn(
            "Binary classification should always be exclusive. Setting "
            "`exclusive_classes` parameter to True"
        )
        exclusive_classes = True

    template = Template(
        load_template(prompt_path).render(
            exclusive_classes=exclusive_classes, labels=labels
        )
    )

    # Create OpenAISuggester with ChatGPT parameters
    openai = TextCatOpenAISuggester(
        prompt_template=template,
        labels=labels,
        max_examples=max_examples,
        segment=segment,
        openai_api_org=api_org,
        openai_api_key=api_key,
        openai_model=model,
        openai_timeout_s=10,
        openai_n=10,
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
    prompt_path=("Path to jinja2 prompt template", "positional", None, Path),
    output_path=("Path to save the output", "positional", None, Path),
    labels=("Labels (comma delimited)", "option", "L", lambda s: s.split(",")),
    lang=("Language to use for tokenizer.", "option", "l", str),
    model=("GPT-3 model to use for completion", "option", "m", str),
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
    prompt_path: Path,
    output_path: Path,
    labels: List[str] = None,
    lang: str = "en",
    model: str = "text-davinci-003",
    batch_size: int = 10,
    segment: bool = False,
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

    # Create partial render of the template
    if not exclusive_classes and len(labels) == 1:
        msg.warn(
            "Binary classification should always be exclusive. Setting "
            "`exclusive_classes` parameter to True"
        )
        exclusive_classes = True

    template = Template(
        load_template(prompt_path).render(
            exclusive_classes=exclusive_classes,
            labels=labels,
        )
    )

    # Create OpenAISuggester with ChatGPT parameters
    openai = TextCatOpenAISuggester(
        prompt_template=template,
        labels=labels,
        max_examples=max_examples,
        segment=segment,
        openai_api_org=api_org,
        openai_api_key=api_key,
        openai_model=model,
        openai_timeout_s=10,
        openai_n=10,
        verbose=verbose,
    )
    for eg in examples:
        openai.add_example(eg)

    # Set up the stream
    stream = list(srsly.read_jsonl(input_path))
    stream = openai(tqdm.tqdm(stream), batch_size=batch_size, nlp=nlp)
    srsly.write_jsonl(output_path, stream)
