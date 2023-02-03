from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, cast, Callable
from dataclasses import dataclass

import prodigy
import prodigy.components.db
import prodigy.components.preprocess
import prodigy.util
import spacy
import srsly
import tqdm
from dotenv import load_dotenv
from spacy.language import Language
from spacy.util import filter_spans

from recipes.openai import OpenAISuggester, PromptExample, get_api_credentials
from recipes.openai import load_template, normalize_label
from recipes.openai import read_prompt_examples

CSS_FILE_PATH = Path(__file__).parent / "style.css"
DEFAULT_PROMPT_PATH = Path(__file__).parent.parent / "templates" / "ner_prompt.jinja2"
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
class NERPromptExample(PromptExample):
    """An example to be passed into an OpenAI NER prompt"""

    text: str
    entities: Dict[str, List[str]]

    @classmethod
    def from_prodigy(cls, example: Dict, labels: Iterable[str]) -> "PromptExample":
        """Create a prompt example from Prodigy's format.
        Only entities with a label from the given set will be retained.
        The given set of labels is assumed to be already normalized.
        """
        if "text" not in example:
            raise ValueError("Cannot make PromptExample without text")
        entities_by_label = defaultdict(list)
        full_text = example["text"]
        for span in example.get("spans", []):
            label = normalize_label(span["label"])
            if label in labels:
                mention = full_text[int(span["start"]) : int(span["end"])]
                entities_by_label[label].append(mention)

        return cls(text=full_text, entities=entities_by_label)


def make_ner_response_parser(labels: List[str], lang: str) -> Callable:

    nlp = spacy.blank(lang)

    def _parse_response(text: str, example: Optional[Dict] = None) -> Dict:
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
                label = normalize_label(label)
                if label in labels:
                    if phrases.strip():
                        phrases = [
                            phrase.strip() for phrase in phrases.strip().split(",")
                        ]
                        output.append((label, phrases))

        example = _fmt_response(output, example)
        return example

    def _fmt_response(response: List[Tuple[str, List[str]]], example: Dict):
        doc = nlp.make_doc(example["text"])
        spacy_spans = []
        for label, phrases in response:
            label = normalize_label(label)
            if label in labels:
                offsets = _find_substrings(doc.text, phrases)
                for start, end in offsets:
                    span = doc.char_span(
                        start, end, alignment_mode="contract", label=label
                    )
                    if span is not None:
                        spacy_spans.append(span)
        # This step prevents the same token from being used in multiple spans.
        # If there's a conflict, the longer span is preserved.
        spacy_spans = filter_spans(spacy_spans)
        spans = [
            {
                "label": span.label_,
                "start": span.start_char,
                "end": span.end_char,
                "token_start": span.start,
                "token_end": span.end - 1,
            }
            for span in spacy_spans
        ]
        return {"spans": spans}

    return _parse_response


@prodigy.recipe(
    "ner.openai.correct",
    dataset=("Dataset to save answers to", "positional", None, str),
    filepath=("Path to jsonl data to annotate", "positional", None, Path),
    labels=("Labels (comma delimited)", "positional", None, lambda s: s.split(",")),
    model=("GPT-3 model to use for initial predictions", "option", "m", str),
    examples_path=("Path to examples to help define the task", "option", "e", Path),
    lang=("Language to use for tokenizer", "option", "l", str),
    max_examples=("Max examples to include in prompt", "option", "n", int),
    prompt_path=("Path to jinja2 prompt template", "option", "p", Path),
    batch_size=("Batch size to send to OpenAI API", "option", "b", int),
    segment=("Split articles into sentences", "flag", "S", bool),
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
    examples = read_prompt_examples(examples_path, example_class=NERPromptExample)
    nlp = spacy.blank(lang)
    if segment:
        nlp.add_pipe("sentencizer")
    api_key, api_org = get_api_credentials(model)
    openai = OpenAISuggester(
        response_parser=make_ner_response_parser(labels=labels, lang=lang),
        openai_model=model,
        labels=labels,
        max_examples=max_examples,
        prompt_template=load_template(prompt_path),
        segment=segment,
        verbose=verbose,
        openai_api_org=api_org,
        openai_api_key=api_key,
        openai_n=1,
        openai_retry_timeout_s=10,
        openai_read_timeout_s=20,
        openai_n_retries=10,
        prompt_example_class=NERPromptExample,
    )
    for eg in examples:
        openai.add_example(eg)
    if max_examples >= 1:
        db = prodigy.components.db.connect()
        db_examples = db.get_dataset(dataset)
        if db_examples:
            for eg in db_examples:
                if PromptExample.is_flagged(eg):
                    openai.add_example(PromptExample.from_prodigy(eg, openai.labels))
    stream = cast(Iterable[Dict], srsly.read_jsonl(filepath))
    return {
        "dataset": dataset,
        "view_id": "blocks",
        "stream": openai(stream, batch_size=batch_size, nlp=nlp),
        "update": openai.update,
        "config": {
            "labels": openai.labels,
            "batch_size": batch_size,
            "exclude_by": "input",
            "blocks": [
                {"view_id": "ner_manual"},
                {"view_id": "html", "html_template": HTML_TEMPLATE},
            ],
            "show_flag": True,
            "global_css": CSS_FILE_PATH.read_text(),
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
    segment: bool = False,
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
    api_key, api_org = get_api_credentials(model)
    examples = read_prompt_examples(examples_path, example_class=NERPromptExample)
    nlp = spacy.blank(lang)
    if segment:
        nlp.add_pipe("sentencizer")
    openai = OpenAISuggester(
        response_parser=make_ner_response_parser(labels=labels, lang=lang),
        openai_model=model,
        labels=labels,
        max_examples=max_examples,
        prompt_template=load_template(prompt_path),
        segment=segment,
        verbose=verbose,
        openai_api_org=api_org,
        openai_api_key=api_key,
        openai_n=1,
        openai_retry_timeout_s=10,
        openai_read_timeout_s=20,
        openai_n_retries=10,
        prompt_example_class=NERPromptExample,
    )
    for eg in examples:
        openai.add_example(eg)
    stream = list(srsly.read_jsonl(input_path))
    stream = openai(tqdm.tqdm(stream), batch_size=batch_size, nlp=nlp)
    srsly.write_jsonl(output_path, stream)


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
    substrings = _unique(substrings)
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


def _unique(items: List[str]) -> List[str]:
    """Remove duplicates without changing order"""
    seen = set()
    output = []
    for item in items:
        if item not in seen:
            output.append(item)
            seen.add(item)
    return output
