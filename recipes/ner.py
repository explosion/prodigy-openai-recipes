import copy
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple, TypeVar, cast, Optional
import time
import sys
import tqdm

import httpx
import spacy
import srsly
from dotenv import load_dotenv
from spacy.language import Language
import prodigy
import prodigy.components.preprocess
import prodigy.util
import prodigy.components.db
import jinja2
import rich
from rich.panel import Panel

_ItemT = TypeVar("_ItemT")

DEFAULT_PROMPT_PATH = Path(__file__).parent.parent / "templates" / "ner_prompt.jinja2"

# Set up openai
load_dotenv()  # take environment variables from .env.


class OpenAISuggester:
    prompt_template: jinja2.Template
    model: str
    labels: List[str]
    max_examples: int
    verbose: bool
    examples: List[Dict]

    def __init__(
        self,
        prompt_template: jinja2.Template,
        model: str,
        labels: List[str],
        max_examples: int,
        verbose: bool = False,
    ):
        self.prompt_template = prompt_template
        self.model = model
        self.labels = labels
        self.max_examples = max_examples
        self.verbose = verbose
        self.examples = []

    def __call__(
        self, stream: Iterable[Dict], *, nlp: Language, batch_size: int
    ) -> Iterable[Dict]:
        stream = self.stream_suggestions(stream, batch_size=batch_size)
        stream = self.format_suggestions(stream, nlp=nlp)
        return stream

    def update(self, examples: Iterable[Dict]) -> float:
        for eg in examples:
            if eg.get("flagged"):
                self.add_example(eg)
        return 0.0

    def add_example(self, example: Dict) -> None:
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
        stream = prodigy.components.preprocess.add_tokens(nlp, stream, skip=True)  # type: ignore
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
        self, text: str, labels: List[str], examples: List[Dict]
    ) -> str:
        """Generate a prompt for named entity annotation.

        The prompt can use examples to further clarify the task. Note that using too
        many examples will make the prompt too large, slowing things down.
        """
        return self.prompt_template.render(text=text, labels=labels, examples=examples)

    def _get_ner_response(self, prompts: List[str]) -> List[str]:
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENAI_KEY')}",
            "OpenAI-Organization": os.getenv("OPENAI_ORG"),
            "Content-Type": "application/json",
        }
        r = _retry429(
            lambda: httpx.post(
                "https://api.openai.com/v1/completions",
                headers=headers,
                json={
                    "model": self.model,
                    "prompt": prompts,
                    "temperature": 0,
                    "max_tokens": 500,
                },
            ),
            n=1,
            timeout_s=1,
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
                    output.append((label, phrases.strip().split(",")))
        return output


@prodigy.recipe(
    "ner.openai.correct",
    dataset=("Dataset to save answers to", "positional", None, str),
    filepath=("Path to jsonl data to annotate", "positional", None, Path),
    labels=("Labels (comma delimited)", "positional", None, lambda s: s.split(",")),
    model=("GPT-3 model to use for completion", "option", "m", str),
    examples=(
        "Path to examples to help define the task",
        "option",
        "e",
        Path,
    ),
    lang=("Language to use for tokenizer", "option", "l", str),
    max_examples=("Max examples to include in prompt", "option", "n", int),
    prompt_path=("Path to jinja2 prompt template", "option", "p", Path),
    batch_size=("Batch size to send to OpenAI API", "option", "b", int),
    verbose=("Print extra information to terminal", "option", "v", bool),
)
def ner_openai_correct(
    dataset: str,
    filepath: Path,
    labels: List[str],
    lang: str = "en",
    model: str = "text-davinci-003",
    batch_size: int = 10,
    examples_path: Optional[Path] = None,
    prompt_path: Path = DEFAULT_PROMPT_PATH,
    max_examples: int = 2,
    verbose: bool = False,
):
    examples = _read_examples(examples_path)
    nlp = spacy.blank(lang)
    openai = OpenAISuggester(
        model=model,
        labels=labels,
        max_examples=max_examples,
        prompt_template=_load_template(prompt_path),
        verbose=verbose,
    )
    for eg in examples:
        openai.add_example(eg)
    if max_examples >= 1:
        db = prodigy.components.db.connect()
        for eg in db.get_dataset(dataset):
            if eg.get("flagged"):
                openai.add_example(eg)
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
    examples=("Examples file to help define the task", "option", "e", Path),
    max_examples=("Max examples to include in prompt", "option", "n", int),
    prompt_path=("Path to jinja2 prompt template", "option", "p", Path),
    batch_size=("Batch size to send to OpenAI API", "option", "b", int),
    verbose=("Print extra information to terminal", "option", "v", bool),
)
def ner_openai_fetch(
    input_path: Path,
    output_path: Path,
    labels: List[str],
    lang: str = "en",
    model: str = "text-davinci-003",
    batch_size: int = 10,
    examples_path: Optional[Path] = None,
    prompt_path: Path = DEFAULT_PROMPT_PATH,
    max_examples: int = 2,
    verbose: bool = False,
):
    examples = _read_examples(examples_path)
    nlp = spacy.blank(lang)
    openai = OpenAISuggester(
        model=model,
        labels=labels,
        max_examples=max_examples,
        prompt_template=_load_template(prompt_path),
        verbose=verbose,
    )
    for eg in examples:
        openai.add_example(eg)
    stream = list(srsly.read_jsonl(input_path))
    stream = openai(tqdm.tqdm(stream), batch_size=batch_size, nlp=nlp)
    srsly.write_jsonl(output_path, stream)


def _read_examples(path: Optional[Path]) -> List[Dict]:
    if path is None:
        return []
    elif path.suffix in (".yml", ".yaml"):
        return _read_yaml_examples(path)
    elif path.suffix == ".json":
        return cast(List[Dict], srsly.read_json(path))
    else:
        # TODO: Improve this
        print("Unknown file format for examples.")
        sys.exit(1)


def _load_template(path: Path) -> jinja2.Template:
    # I know jinja has a lot of complex file loading stuff,
    # but we're not using the inheritence etc that makes
    # that stuff worthwhile.
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


def _read_yaml_examples(path: Path) -> List[Dict]:
    data = srsly.read_yaml(path)
    assert isinstance(data, list)
    output = []
    for item in data:
        output.append({"text": item["text"], "entities": item["entities"].items()})
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
    text: str, substrings: List[str], *, case_sensitive=False
) -> List[Tuple[int, int]]:
    """Given a list of substrings, find their character start and end positions in a text. The substrings are assumed to be sorted by the order of their occurrence in the text."""
    if not case_sensitive:
        text = text.lower()
        substrings = [s.lower() for s in substrings]
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


def test_template_no_examples():
    text = "David Bowie lived in Berlin in the 1970s."
    labels = ["PERSON", "PLACE", "PERIOD"]
    examples = []
    path = Path(__file__).parent.parent / "templates" / "ner_prompt.jinja2"
    template = _load_template(path)
    prompt = template.render(text=text, labels=labels, examples=examples)
    assert (
        prompt
        == f"""
From the text below, extract the following entities in the following format:
PERSON: <comma delimited list of strings>
PLACE: <comma delimited list of strings>
PERIOD: <comma delimited list of strings>

Text:
\"\"\"
David Bowie lived in Berlin in the 1970s.
\"\"\"
""".lstrip()
    )


def test_template_two_examples():
    text = "David Bowie lived in Berlin in the 1970s."
    labels = ["PERSON", "PLACE", "PERIOD"]
    examples = [
        {"text": "New York is a large city.", "entities": [["PLACE", ["New York"]]]},
        {
            "text": "David Hasslehoff and Helena Fischer are big in Germany.",
            "entities": [
                ["PERSON", ["David Hasslehoff", "Helena Fischer"]],
                ["PLACE", ["Germany"]],
            ],
        },
    ]
    path = Path(__file__).parent.parent / "templates" / "ner_prompt.jinja2"
    template = _load_template(path)
    prompt = template.render(text=text, labels=labels, examples=examples)
    assert (
        prompt
        == f"""
From the text below, extract the following entities in the following format:
PERSON: <comma delimited list of strings>
PLACE: <comma delimited list of strings>
PERIOD: <comma delimited list of strings>

Text:
\"\"\"
David Bowie lived in Berlin in the 1970s.
\"\"\"

For example:

Text:
\"\"\"
New York is a large city.
\"\"\"
PLACE: New York

Text:
\"\"\"
David Hasslehoff and Helena Fischer are big in Germany.
\"\"\"
PERSON: David Hasslehoff, Helena Fischer
PLACE: Germany

""".lstrip()
    )
