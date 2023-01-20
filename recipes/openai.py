import abc
import copy
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, TypeVar

import httpx
import jinja2
import prodigy
import rich
import srsly
from prodigy.components import preprocess
from prodigy.util import msg, set_hashes
from rich.panel import Panel
from spacy.language import Language

_ItemT = TypeVar("_ItemT")


@dataclass
class PromptExample(abc.ABC):
    """An example to be passed into an OpenAI prompt."""

    @staticmethod
    def is_flagged(example: Dict) -> bool:
        """Check whether a Prodigy example is flagged for use
        in the prompt."""

        return (
            example.get("flagged") is True
            and example.get("answer") == "accept"
            and "text" in example
        )

    @abc.abstractclassmethod
    def from_prodigy(cls, example: Dict, labels: Iterable[str]) -> "PromptExample":
        """Create a prompt example from Prodigy's format."""
        pass


def normalize_label(label: str) -> str:
    return label.lower()


class OpenAISuggester:
    """Suggest annotations using OpenAI's ChatGPT"""

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
    openai_read_timeout_s: int
    openai_n: int
    examples: List[PromptExample]
    response_parser: Callable

    def __init__(
        self,
        prompt_template: jinja2.Template,
        *,
        labels: List[str],
        max_examples: int,
        segment: bool,
        openai_api_org: str,
        openai_api_key: str,
        openai_model: str,
        response_parser: Callable,
        openai_temperature: int = 0,
        openai_max_tokens: int = 500,
        openai_timeout_s: int = 1,
        openai_read_timeout_s: int = 30,
        openai_n: int = 1,
        verbose: bool = False,
    ):
        self.prompt_template = prompt_template
        self.model = openai_model
        self.labels = [normalize_label(label) for label in labels]
        self.max_examples = max_examples
        self.verbose = verbose
        self.segment = segment
        self.examples = []
        self.openai_api_org = openai_api_org
        self.openai_api_key = openai_api_key
        self.openai_temperature = openai_temperature
        self.openai_max_tokens = openai_max_tokens
        self.openai_timeout_s = openai_timeout_s
        self.openai_read_timeout_s = openai_read_timeout_s
        self.openai_n = openai_n
        self.response_parser = response_parser

    def __call__(
        self,
        stream: Iterable[Dict],
        *,
        nlp: Language,
        batch_size: int,
        **kwargs,
    ) -> Iterable[Dict]:
        if self.segment:
            stream = preprocess.split_sentences(nlp, stream)

        stream = self.pipe(stream, nlp, batch_size, **kwargs)
        stream = self.set_hashes(stream)
        return stream

    def pipe(self, stream: Iterable[Dict], nlp: Language, batch_size: int, **kwargs):
        """Process the stream and add suggestions from OpenAI."""
        stream = self.stream_suggestions(stream, batch_size)
        stream = self.format_suggestions(stream, nlp=nlp)
        return stream

    def set_hashes(self, stream: Iterable[Dict]):
        for example in stream:
            yield set_hashes(example)

    def update(self, examples: Iterable[Dict]) -> float:
        """Update the examples that will be used in the prompt based on user flags."""
        for eg in examples:
            if PromptExample.is_flagged(eg):
                self.add_example(PromptExample.from_prodigy(eg, self.label))
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
        """Get zero-shot or few-shot annotations from OpenAI.

        Given a stream of input examples, we define a prompt, get a response from OpenAI,
        and yield each example with their predictions to the output stream.
        """
        for batch in batch_sequence(stream, batch_size):
            prompts = [
                self._get_prompt(eg["text"], labels=self.labels, examples=self.examples)
                for eg in batch
            ]
            responses = self._get_openai_response(prompts)
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
            if "meta" not in example:
                example["meta"] = {}

            response = example["openai"].get("response", "")
            example.update(self.response_parser(response=response))
            yield example

    def _get_prompt(
        self, text: str, labels: List[str], examples: List[PromptExample]
    ) -> str:
        """Generate a prompt for ChatGPT OpenAI."""
        return self.prompt_template.render(text=text, labels=labels, examples=examples)

    def _get_openai_response(self, prompts: List[str]) -> List[str]:
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "OpenAI-Organization": self.openai_api_org,
            "Content-Type": "application/json",
        }
        r = retry(
            lambda: httpx.post(
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
            timeout_s=self.openai_read_timeout_s,
        )
        r.raise_for_status()
        responses = r.json()
        return [responses["choices"][i]["text"] for i in range(len(prompts))]


def get_api_credentials(model: str = None) -> Tuple[str, str]:
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
    r = retry(
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


def read_prompt_examples(
    path: Optional[Path], *, example_class: PromptExample
) -> List[PromptExample]:
    if path is None:
        return []
    elif path.suffix in (".yml", ".yaml"):
        return read_yaml_examples(path, example_class=example_class)
    elif path.suffix == ".json":
        data = srsly.read_json(path)
        assert isinstance(data, list)
        return [PromptExample(**eg) for eg in data]
    else:
        msg.fail(
            "The --examples-path (-e) parameter expects a .yml, .yaml or .json file."
        )
        sys.exit(-1)


def load_template(path: Path) -> jinja2.Template:
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
    return jinja2.Template(text, undefined=jinja2.DebugUndefined)


def retry(
    call_api: Callable[[], httpx.Response],
    n: int,
    timeout_s: int,
    error_codes: List[int] = [429, 503],
) -> httpx.Response:
    """Retry a call to the OpenAI API if we get a non-ok status code.

    This function automatically retries a request if it catches a response
    with an error code in `error_codes`. The amount of timeout also increases
    exponentially every time we retry.
    """
    assert n >= 0
    assert timeout_s >= 1
    r = call_api()
    i = -1
    # We don't want to retry on every non-ok status code. Some are about
    # incorrect inputs, etc. and we want to terminate on those.
    while i < n and r.status_code in error_codes:
        time.sleep(timeout_s)
        i += 1
        timeout_s = timeout_s * 2  # Increase timeout everytime you retry
        msg.text(
            f"Retrying call (retries left: {n-i}, timeout: {timeout_s}s). "
            f"Previous call returned: {r.status_code} ({r.reason_phrase})"
        )
    return r


def read_yaml_examples(
    path: Path, *, example_class: PromptExample
) -> List[PromptExample]:
    data = srsly.read_yaml(path)
    if not isinstance(data, list):
        msg.fail("Cannot interpret prompt examples from yaml", exits=True)
    assert isinstance(data, list)
    output = [example_class(**eg) for eg in data]
    return output


def batch_sequence(items: Iterable[_ItemT], batch_size: int) -> Iterable[List[_ItemT]]:
    batch = []
    for eg in items:
        batch.append(eg)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
