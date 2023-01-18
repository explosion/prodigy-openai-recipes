from pathlib import Path
from typing import Optional, Dict, List

import prodigy
import spacy
import srsly
import tqdm
from dotenv import load_dotenv
from prodigy.util import msg

from .openai import OpenAISuggester, get_api_credentials, load_template
from .openai import read_prompt_examples

DEFAULT_LABELS = ["PER", "ORG", "LOC"]
CSS_FILE_PATH = Path(__file__).parent / "style.css"
HTML_TEMPLATE = """
<div class="cleaned">
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


class TextCatOpenAISuggester(OpenAISuggester):
    def parse_response(self, example: Dict, response: str) -> Dict:
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
        example["answer"] = output["answer"] == "accept"
        example["meta"]["reason"] = output["reason"]
        example["label"] = self.labels[0]
        return example

    def _parse_multi(self, example: Dict, response: str) -> Dict:
        """Parse multilabel TextCat."""
        output = self._parse_output(response)
        example["options"] = [{"id": label, "text": label} for label in self.labels]
        example["meta"]["reason"] = output["reason"]
        example["answer"] = "accept"
        example["accept"] = [s.strip() for s in output["answer"].split(",")]
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
    labels: List[str] = DEFAULT_LABELS,
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
    examples = read_prompt_examples(examples_path)
    if labels is None:
        msg.fail("textcat.teach requires at least one --label", exits=1)
    nlp = spacy.blank(lang)

    if segment:
        nlp.add_pipe("sentencizer")

    # Create OpenAISuggester with ChatGPT parameters
    openai = TextCatOpenAISuggester(
        prompt_template=load_template(prompt_path),
        labels=labels,
        max_examples=max_examples,
        segment=segment,
        open_api_org=api_org,
        open_api_key=api_key,
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
    if not exclusive_classes and len(labels) == 1:
        msg.warn(
            "Binary classification should always be exclusive. Setting "
            "`exclusive_classes` parameter to True"
        )
        exclusive_classes = True

    return {
        "dataset": dataset,
        "view_id": "blocks",
        "stream": stream,
        "update": openai.update,
        "config": {
            "labels": openai.labels[0] if len(labels) == 1 else openai.labels,
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
