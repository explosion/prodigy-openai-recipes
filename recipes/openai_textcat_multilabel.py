import copy
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, TypeVar

import httpx
import jinja2
import prodigy
import prodigy.components.db
import prodigy.components.preprocess
import prodigy.components.sorters
import prodigy.models.textcat
import prodigy.types
import prodigy.util
import rich
import spacy
import srsly
import tqdm
from dotenv import load_dotenv
from prodigy.util import log, msg
from rich.panel import Panel
from spacy.language import Language

timeout = httpx.Timeout(120.0, connect=60.0)
client = httpx.Client(timeout=timeout)

_ItemT = TypeVar("_ItemT")

DEFAULT_PROMPT_PATH = (
    Path(__file__).parent.parent / "templates" / "textcat_multilabel_prompt.jinja2"
)
CSS_FILE_PATH = Path(__file__).parent / "style.css"

TEXTCAT_LABELS = ["Recipe", "Comment", "Story"]

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


@dataclass
class PromptExample:
    """An example to be passed into an OpenAI TextCat prompt."""

    text: str
    labels: list = field(default_factory=list)

    @staticmethod
    def is_flagged(example: Dict) -> bool:
        """Check whether a Prodigy example is flagged for use
        in the prompt."""

        return (
            example.get("flagged") is True
            and example.get("answer") == "accept"
            and "text" in example
        )

    @classmethod
    def from_prodigy(cls, example: Dict, labels: Iterable[str]) -> "PromptExample":
        """Create a prompt example from Prodigy's format.
        Only entities with a label from the given set will be retained.
        The given set of labels is assumed to be already normalized.
        """
        if "text" not in example:
            raise ValueError("Cannot make PromptExample without text")

        full_text = example["text"]
        label = example["label"]
        return cls(text=full_text, label=_normalize_label(label))


def _normalize_label(label: str) -> str:
    return label.lower()


class OpenAISuggester:
    prompt_template: jinja2.Template
    model: str
    labels: List[str]
    max_examples: int
    segment: bool
    verbose: bool
    openai_api_org: str
    openai_api_key: str
    openai_temperature: int
    openai_max_tokens: int
    openai_timeout_s: int
    openai_n: int
    examples: List[PromptExample]

    def __init__(
        self,
        prompt_template: jinja2.Template,
        *,
        labels: List[str],
        max_examples: int,
        segment: bool,
        exclusive_classes: bool,
        openai_api_org: str,
        openai_api_key: str,
        openai_model: str,
        openai_temperature: int = 0,
        openai_max_tokens: int = 500,
        openai_timeout_s: int = 1,
        openai_n: int = 1,
        verbose: bool = False,
    ):
        self.prompt_template = prompt_template
        self.model = openai_model
        self.labels = [_normalize_label(label) for label in labels]
        self.max_examples = max_examples
        self.verbose = verbose
        self.segment = segment
        self.examples = []
        self.exclusive_classes = exclusive_classes
        self.openai_api_org = openai_api_org
        self.openai_api_key = openai_api_key
        self.openai_temperature = openai_temperature
        self.openai_max_tokens = openai_max_tokens
        self.openai_timeout_s = openai_timeout_s
        self.openai_n = openai_n

    def __call__(
        self,
        stream: Iterable[Dict],
        *,
        nlp: Language,
        batch_size: int,
    ) -> Iterable[Dict]:
        if self.segment:
            stream = prodigy.components.preprocess.split_sentences(nlp, stream)

        stream = self.stream_suggestions(stream, batch_size=batch_size)
        stream = self.format_suggestions(stream, nlp=nlp)
        return stream

    def update(self, examples: Iterable[Dict]) -> float:
        for eg in examples:
            if PromptExample.is_flagged(eg):
                self.add_example(PromptExample.from_prodigy(eg, self.labels))
        return 0.0

    def add_example(self, example: PromptExample) -> None:
        """Add an example for use in the prompts. Examples are pruned to the most recent max_examples."""
        if self.max_examples:
            self.examples.append(example)
        if len(self.examples) >= self.max_examples:
            self.examples = self.examples[-self.max_examples :]

    def stream_suggestions(
        self, stream: Iterable[Dict], batch_size: int
    ) -> Iterable[Dict]:
        """Get zero-shot or few-shot suggested TextCat annotations from OpenAI.

        Given a stream of input examples, we define a prompt, get a response from OpenAI,
        and yield each example with their predictions to the output stream.
        """
        for batch in _batch_sequence(stream, batch_size):
            prompts = [
                self._get_textcat_prompt(
                    eg["text"], labels=self.labels, examples=self.examples
                )
                for eg in batch
            ]
            responses = self._get_textcat_response(prompts, delay=0)
            for eg, prompt, response in zip(batch, prompts, responses):
                if self.verbose:
                    rich.print(Panel(prompt, title="Prompt to OpenAI"))
                eg["openai"] = {"prompt": prompt, "response": response}
                if self.verbose:
                    rich.print(Panel(response, title="Response from OpenAI"))
                yield eg

    def format_suggestions(
        self, stream: Iterable[Dict], *, nlp: Language
    ) -> Iterable[Dict]:
        """Parse the examples in the stream and set up labels
        to display in the Prodigy UI.
        """
        stream = prodigy.components.preprocess.add_tokens(nlp, stream, skip=True)  # type: ignore
        for example in stream:
            example = copy.deepcopy(example)
            response = self._parse_response(example["openai"]["response"])
            # Format the actual task
            example["options"] = [{"id": label, "text": label} for label in self.labels]
            example["meta"]["reason"] = response["reason"]
            example["answer"] = "accept"
            example["accept"] = [s.strip() for s in response["answer"].split(",")]

            yield prodigy.util.set_hashes(example)

    def _get_textcat_prompt(
        self, text: str, labels: List[str], examples: List[PromptExample]
    ) -> str:
        """Generate a prompt for multilabel text categorization."""
        prompt = self.prompt_template.render(
            text=text, labels=labels, exclusive=self.exclusive_classes
        )
        return prompt

    def _get_textcat_response(self, prompts: List[str], delay: int = 0) -> List[str]:
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "OpenAI-Organization": self.openai_api_org,
            "Content-Type": "application/json",
        }
        time.sleep(delay)
        r = _retry429(
            lambda: client.post(
                "https://api.openai.com/v1/completions",
                headers=headers,
                json={
                    "model": self.model,
                    "prompt": prompts,
                    "temperature": self.openai_temperature,
                    "max_tokens": self.openai_max_tokens,
                },
            ),
            n=self.openai_n,
            timeout_s=self.openai_timeout_s,
        )
        r.raise_for_status()
        responses = r.json()
        return [responses["choices"][i]["text"] for i in range(len(prompts))]

    def _parse_response(self, text: str) -> Dict[str, str]:
        """Interpret OpenAI's TextCat response. It's supposed to be
        a list of lines, with each line having the form:

        answer: <accept/reject for binary or the label for multilabel>
        reason: <string>

        However, there's no guarantee that the model will give
        us well-formed output. It could say anything, it's an LM.
        So we need to be robust.
        """
        output = {}
        for line in text.strip().split("\n"):
            if line and ":" in line:
                key, value = line.split(":", 1)
                output[key.strip().lower()] = value.strip()
        return output


@prodigy.recipe(
    # fmt: off
    "textcat_multilabel.openai.correct",
    input_path=("Path to jsonl data to annotate", "positional", None, Path),
    labels=("Labels (comma delimited)", "positional", None, lambda s: s.split(",")),
    lang=("Language to initialize spaCy model", "option", "l", str),
    model=("GPT-3 model to use for completion", "option", "m", str),
    batch_size=("Batch size to send to OpenAI API", "option", "b", int),
    segment=("Split sentences", "flag", "S", bool),
    examples_path=("Examples file to help define the task", "option", "e", Path),
    prompt_path=("Path to jinja2 prompt template", "option", "p", Path),
    max_examples=("Max examples to include in prompt", "option", "n", int),
    exclusive_classes=("Make the classification task exclusive", "flag", "E", bool),
    verbose=("Print extra information to terminal", "flag", "v", bool),
    # fmt: on
)
def textcat_multilabel_openai_correct(
    dataset: str,
    input_path: Path,
    labels: List[str],
    lang: str = "en",
    model: str = "text-davinci-003",
    batch_size: int = 10,
    segment: bool = False,
    examples_path: Optional[Path] = None,
    prompt_path: Path = DEFAULT_PROMPT_PATH,
    max_examples: int = 2,
    exclusive_classes: bool = False,
    verbose: bool = False,
):
    """Get bulk TextCat suggestions from an OpenAI API, using zero-shot or
    few-shot learning.  The results can then be corrected using the
    `textcat.manual` recipe.

    Here, we use ChatGPT to suggest if a particular text talks about a recipe or
    not. You can use the parameter `--chatgpt-bias` (float, 0 to 1 inclusive) to
    set how much we prefer seeing examples that ChatGPT thinks as recipes.
    """

    api_key, api_org = _get_api_credentials(model)
    examples = _read_prompt_examples(examples_path)
    if labels is None:
        msg.fail("textcat.teach requires at least one --label", exits=1)
    nlp = spacy.blank(lang)

    if segment:
        nlp.add_pipe("sentencizer")

    openai = OpenAISuggester(
        openai_model=model,
        labels=labels,
        max_examples=max_examples,
        prompt_template=_load_template(prompt_path),
        exclusive_classes=exclusive_classes,
        verbose=verbose,
        segment=segment,
        openai_api_key=api_key,
        openai_api_org=api_org,
        openai_timeout_s=120,
        openai_n=10,
    )

    for eg in examples:
        openai.add_example(eg)

    stream = list(srsly.read_jsonl(input_path))
    stream = openai(tqdm.tqdm(stream), batch_size=batch_size, nlp=nlp)

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
                {"view_id": "choice"},
                {"view_id": "html", "html_template": HTML_TEMPLATE},
            ],
            "show_flag": True,
            "global_css": CSS_FILE_PATH.read_text(),
        },
    }


@prodigy.recipe(
    "textcat_multilabel.openai.fetch",
    input_path=("Path to jsonl data to annotate", "positional", None, Path),
    output_path=("Path to save the output", "positional", None, Path),
    labels=("Labels (comma delimited)", "positional", None, lambda s: s.split(",")),
    lang=("Language to use for tokenizer.", "option", "l", str),
    model=("GPT-3 model to use for completion", "option", "m", str),
    examples_path=("Examples file to help define the task", "option", "e", Path),
    max_examples=("Max examples to include in prompt", "option", "n", int),
    prompt_path=("Path to jinja2 prompt template", "option", "p", Path),
    batch_size=("Batch size to send to OpenAI API", "option", "b", int),
    segment=("Split sentences", "flag", "S", bool),
    verbose=("Print extra information to terminal", "option", "flag", bool),
)
def textcat_multilabel_openai_fetch(
    input_path: Path,
    output_path: Path,
    labels: List[str],
    lang: str = "en",
    model: str = "text-davinci-003",
    batch_size: int = 10,
    segment: bool = False,
    examples_path: Optional[Path] = None,
    prompt_path: Path = DEFAULT_PROMPT_PATH,
    max_examples: int = 2,
    verbose: bool = False,
):
    """Get bulk TextCat suggestions from an OpenAI API, using zero-shot or few-shot learning.
    The results can then be corrected using the `textcat.manual` recipe."""
    api_key, api_org = _get_api_credentials(model)
    examples = _read_prompt_examples(examples_path)
    nlp = spacy.blank(lang)
    if segment:
        nlp.add_pipe("sentencizer")
    openai = OpenAISuggester(
        openai_model=model,
        labels=labels,
        max_examples=max_examples,
        prompt_template=_load_template(prompt_path),
        verbose=verbose,
        segment=segment,
        openai_api_key=api_key,
        openai_api_org=api_org,
        openai_timeout_s=120,
        openai_n=10,
    )
    for eg in examples:
        openai.add_example(eg)
    stream = list(srsly.read_jsonl(input_path))
    stream = openai(tqdm.tqdm(stream), batch_size=batch_size, nlp=nlp)
    srsly.write_jsonl(output_path, stream)


def _get_api_credentials(model: str = None) -> Tuple[str, str]:
    # Fetch and check the key
    api_key = os.getenv("OPENAI_KEY")
    if api_key is None:
        m = (
            "Could not find the API key to access the openai API. Ensure you have an API key "
            "set up via https://beta.openai.com/account/api-keys, then make it available as "
            "an environment variable 'OPENAI_KEY', for instance in a .env file."
        )
        msg.fail(m)
        sys.exit(-1)
    # Fetch and check the org
    org = os.getenv("OPENAI_ORG")
    if org is None:
        m = (
            "Could not find the organisation to access the openai API. Ensure you have an API key "
            "set up via https://beta.openai.com/account/api-keys, obtain its organization ID 'org-XXX' "
            "via https://beta.openai.com/account/org-settings, then make it available as "
            "an environment variable 'OPENAI_ORG', for instance in a .env file."
        )
        msg.fail(m)
        sys.exit(-1)

    # Check the access and get a list of available models to verify the model argument (if not None)
    # Even if the model is None, this call is used as a healthcheck to verify access.
    headers = {
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Organization": org,
    }
    r = _retry429(
        lambda: httpx.get(
            "https://api.openai.com/v1/models",
            headers=headers,
        ),
        n=1,
        timeout_s=1,
    )
    if r.status_code == 422:
        m = (
            "Could not access api.openai.com -- 422 permission denied."
            "Visit https://beta.openai.com/account/api-keys to check your API keys."
        )
        msg.fail(m)
        sys.exit(-1)
    elif r.status_code != 200:
        m = "Error accessing api.openai.com" f"{r.status_code}: {r.text}"
        msg.fail(m)
        sys.exit(-1)

    if model is not None:
        response = r.json()["data"]
        models = [response[i]["id"] for i in range(len(response))]
        if model not in models:
            e = f"The specified model '{model}' is not available. Choices are: {sorted(set(models))}"
            msg.fail(e, exits=1)

    return api_key, org


def _read_prompt_examples(path: Optional[Path]) -> List[PromptExample]:
    if path is None:
        return []
    elif path.suffix in (".yml", ".yaml"):
        return _read_yaml_examples(path)
    elif path.suffix == ".json":
        data = srsly.read_json(path)
        assert isinstance(data, list)
        return [PromptExample(**eg) for eg in data]
    else:
        msg.fail(
            "The --examples-path (-e) parameter expects a .yml, .yaml or .json file."
        )
        sys.exit(-1)


def _load_template(path: Path) -> jinja2.Template:
    # I know jinja has a lot of complex file loading stuff,
    # but we're not using the inheritance etc that makes
    # that stuff worthwhile.
    if not path.suffix == ".jinja2":
        msg.fail(
            "The --prompt-path (-p) parameter expects a .jinja2 file.",
            exits=1,
        )
    with path.open("r", encoding="utf8") as file_:
        text = file_.read()
    return jinja2.Template(text)


def _retry429(
    call_api: Callable[[], httpx.Response], n: int, timeout_s: int
) -> httpx.Response:
    """Retry a call to the OpenAI API if we get a 429: Too many requests
    error.
    """
    assert n >= 0
    assert timeout_s >= 1
    r = call_api()
    i = -1
    while i < n and r.status_code == 429:
        time.sleep(timeout_s)
        i += 1
    return r


def _read_yaml_examples(path: Path) -> List[PromptExample]:
    data = srsly.read_yaml(path)
    if not isinstance(data, list):
        msg.fail("Cannot interpret prompt examples from yaml", exits=True)
    assert isinstance(data, list)
    output = []
    for item in data:
        output.append(PromptExample(text=item["text"], entities=item["entities"]))
    return output


def _batch_sequence(items: Iterable[_ItemT], batch_size: int) -> Iterable[List[_ItemT]]:
    batch = []
    for eg in items:
        batch.append(eg)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
