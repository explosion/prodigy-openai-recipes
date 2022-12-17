import copy
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple, TypeVar, cast, Optional
import time
import tqdm
import sys
from dataclasses import dataclass
from collections import defaultdict

import httpx
import spacy
import srsly
from dotenv import load_dotenv
from spacy.language import Language
import prodigy
from prodigy.components.preprocess import split_sentences, add_tokens
import prodigy.util
from prodigy.util import msg
import prodigy.components.db
import jinja2
import rich
from rich.panel import Panel

_ItemT = TypeVar("_ItemT")

DEFAULT_PROMPT_PATH = Path(__file__).parent.parent / "templates" / "ner_prompt.jinja2"

# Set up openai
load_dotenv()  # take environment variables from .env.


@dataclass
class PromptExample:
    """An example to be passed into an OpenAI NER prompt."""

    text: str
    entities: Dict[str, List[str]]

    @staticmethod
    def is_flagged(example: Dict) -> bool:
        """Check whether a Prodigy example is flagged for use
        in the prompt."""

        return (
            example.get("flagged") == True
            and example.get("answer") == "accept"
            and "text" in example
        )

    @classmethod
    def from_prodigy(cls, example: Dict) -> "PromptExample":
        """Create a prompt example from Prodigy's format."""
        if "text" not in example:
            raise ValueError("Cannot make PromptExample without text")
        entities_by_label = defaultdict(list)
        full_text = example["text"]
        for span in example.get("spans", []):
            mention = full_text[int(span["start"]) : int(span["end"]) + 1]
            entities_by_label[span["label"]].append(mention)

        return cls(text=full_text, entities=entities_by_label)


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
        self.labels = labels
        self.max_examples = max_examples
        self.verbose = verbose
        self.segment = segment
        self.examples = []
        self.openai_api_org = openai_api_org
        self.openai_api_key = openai_api_key
        self.openai_temperature = openai_temperature
        self.openai_max_tokens = openai_max_tokens
        self.openai_timeout_s = openai_timeout_s
        self.openai_n = openai_n

    def __call__(
        self, stream: Iterable[Dict], *, nlp: Language, batch_size: int
    ) -> Iterable[Dict]:
        if self.segment:
            stream = split_sentences(nlp, stream)

        stream = self.stream_suggestions(stream, batch_size=batch_size)
        stream = self.format_suggestions(stream, nlp=nlp)
        return stream

    def update(self, examples: Iterable[Dict]) -> float:
        for eg in examples:
            if PromptExample.is_flagged(eg):
                self.add_example(PromptExample.from_prodigy(eg))
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
        """Get zero-shot or few-shot suggested NER annotationss from OpenAI.

        Given a stream of example dictionaries, we calculate a
        prompt, get a response from OpenAI, and add them to the
        dictionary. A further function then takes care of parsing
        the response and setting up the span annotations for Prodigy.
        """
        for batch in _batch_sequence(stream, batch_size):
            prompts = [
                self._get_ner_prompt(
                    eg["text"], labels=self.labels, examples=self.examples
                )
                for eg in batch
            ]
            responses = self._get_ner_response(prompts)
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
        stream = add_tokens(nlp, stream, skip=True)  # type: ignore
        for example in stream:
            example = copy.deepcopy(example)
            # This tokenizes the text with spaCy, so that the token boundaries can be used
            # during the annotation. Without the token boundaries, you need to get the
            # annotation exactly on the characters, which is annoying.
            doc = nlp.make_doc(example["text"])
            response = self._parse_response(example["openai"]["response"])
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
            example = prodigy.util.set_hashes({**example, "spans": spans})
            yield example

    def _get_ner_prompt(
        self, text: str, labels: List[str], examples: List[PromptExample]
    ) -> str:
        """Generate a prompt for named entity annotation.

        The prompt can use examples to further clarify the task. Note that using too
        many examples will make the prompt too large, slowing things down.
        """
        return self.prompt_template.render(text=text, labels=labels, examples=examples)

    def _get_ner_response(self, prompts: List[str]) -> List[str]:
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "OpenAI-Organization": self.openai_api_org,
            "Content-Type": "application/json",
        }
        r = _retry429(
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
            timeout_s=self.openai_timeout_s,
        )
        r.raise_for_status()
        responses = r.json()
        return [responses["choices"][i]["text"] for i in range(len(prompts))]

    def _parse_response(self, text: str) -> List[Tuple[str, List[str]]]:
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
                    phrases = [phrase.strip() for phrase in phrases.strip().split(",")]
                    output.append((label, phrases))
        return output


@prodigy.recipe(
    "ner.openai.correct",
    dataset=("Dataset to save answers to", "positional", None, str),
    filepath=("Path to jsonl data to annotate", "positional", None, Path),
    labels=("Labels (comma delimited)", "positional", None, lambda s: s.split(",")),
    model=("GPT-3 model to use for completion", "option", "m", str),
    examples_path=(
        "Path to examples to help define the task",
        "option",
        "e",
        Path,
    ),
    lang=("Language to use for tokenizer", "option", "l", str),
    max_examples=("Max examples to include in prompt", "option", "n", int),
    prompt_path=("Path to jinja2 prompt template", "option", "p", Path),
    batch_size=("Batch size to send to OpenAI API", "option", "b", int),
    segment=("Split sentences", "flag", "U", bool),
    verbose=("Print extra information to terminal", "flag", "v", bool),
)
def ner_openai_correct(
    dataset: str,
    filepath: Path,
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
    examples = _read_prompt_examples(examples_path)
    nlp = spacy.blank(lang)
    if not segment:
        nlp.add_pipe("sentencizer")
    api_key, api_org = _get_api_credentials()
    openai = OpenAISuggester(
        openai_model=model,
        labels=labels,
        max_examples=max_examples,
        prompt_template=_load_template(prompt_path),
        segment=segment,
        verbose=verbose,
        openai_api_org=api_org,
        openai_api_key=api_key,
    )
    for eg in examples:
        openai.add_example(eg)
    if max_examples >= 1:
        db = prodigy.components.db.connect()
        db_examples = db.get_dataset(dataset)
        if db_examples:
            for eg in db_examples:
                if PromptExample.is_flagged(eg):
                    openai.add_example(PromptExample.from_prodigy(eg))
    stream = cast(Iterable[Dict], srsly.read_jsonl(filepath))
    return {
        "dataset": dataset,
        "view_id": "blocks",
        "stream": openai(stream, batch_size=batch_size, nlp=nlp),
        "update": openai.update,
        "config": {
            "labels": labels,
            "batch_size": batch_size,
            "exclude_by": "input",
            "blocks": [{"view_id": "ner_manual"}, {"view_id": "html"}],
            "show_flag": True,
            "global_css": (Path(__file__).parent / "style.css").read_text(),
        },
    }


@prodigy.recipe(
    "ner.openai.fetch",
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
def ner_openai_fetch(
    input_path: Path,
    output_path: Path,
    labels: List[str],
    lang: str = "en",
    model: str = "text-davinci-003",
    batch_size: int = 10,
    segment: bool = True,
    examples_path: Optional[Path] = None,
    prompt_path: Path = DEFAULT_PROMPT_PATH,
    max_examples: int = 2,
    verbose: bool = False,
):
    """Get bulk NER suggestions from an OpenAI API, using zero-shot or few-shot learning.
    The results can then be corrected using the `ner.manual` recipe.

    This approach lets you get the openai queries out of the way upfront, which can help
    if you want to use multiple annotators of if you want to make sure you don't have to
    wait on the OpenAI queries. The downside is that you can't flag examples to be integrated
    into the prompt during the annotation, unlike the ner.openai.correct recipe.
    """
    api_key, api_org = _get_api_credentials()
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
    )
    for eg in examples:
        openai.add_example(eg)
    stream = list(srsly.read_jsonl(input_path))
    stream = openai(tqdm.tqdm(stream), batch_size=batch_size, nlp=nlp)
    srsly.write_jsonl(output_path, stream)


def _get_api_credentials() -> Tuple[str, str]:
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
    # Check the access. It would be nice to have a healthcheck endpoint, but failing that
    # we query for the models. We just want to see that we can reach the API and that we don't
    # get an access denied.
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
    # but we're not using the inheritence etc that makes
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


def _find_substrings(
    text: str,
    substrings: List[str],
    *,
    case_sensitive: bool = False,
    single_match: bool = False,
) -> List[Tuple[int, int]]:
    """Given a list of substrings, find their character start and end positions in a text. The substrings are assumed to be sorted by the order of their occurrence in the text.

    text: The text to search over.
    substrings: The strings to find.
    case_sensitive: Whether to search without case sensitivity.
    single_match: If False, allow one substring to match multiple times in the text. If True, returns the first hit.
    """
    # remove empty and duplicate strings, and lowercase everything if need be
    substrings = [s for s in substrings if s and len(s) > 0]
    if not case_sensitive:
        text = text.lower()
        substrings = [s.lower() for s in substrings]
    substrings = list(set(substrings))
    offsets = []
    for substring in substrings:
        search_from = 0
        # Search until one hit is found. Continue only if single_match is False.
        while True:
            start = text.find(substring, search_from)
            if start == -1:
                break
            end = start + len(substring)
            offsets.append((start, end))
            if single_match:
                break
            search_from = end
    return offsets
