"""Evaluate the accuracy of the OpenAI zero-shot NER suggestions,
when compared to the corrected data.

We want to make sure we use the same responses we used during
annotation, so we read those in from the jsonl. We also read in
the spaCy-formatted evaluation samples.
"""
from typing import Dict, List, Tuple
import typer
import spacy
from dataclasses import dataclass
from spacy.language import Language
from pathlib import Path
from spacy.tokens import DocBin, Doc, Span
from spacy.training import Example
from spacy.scorer import Scorer
from spacy.util import filter_spans
import srsly


@dataclass
class ProdigyNER:
    text: str
    openai_prompt: str
    openai_response: str

    @classmethod
    def from_dict(cls, prodigy_data) -> "ProdigyNER":
        return ProdigyNER(
            text=prodigy_data["text"],
            openai_prompt=prodigy_data["openai"]["prompt"],
            openai_response=prodigy_data["openai"]["response"],
        )


def find_to_evaluate(docs: List[Doc], dataset: List[ProdigyNER]) -> List[ProdigyNER]:
    # Identify the target docs in the dataset by their text content.
    targets = {d.text for d in docs}
    return [x for x in dataset if x.text in targets]


def make_docs(nlp: Language, dataset: List[ProdigyNER]) -> List[Doc]:
    output = []
    for x in dataset:
        doc = nlp.make_doc(x.text)
        spans = _get_openai_spans(doc, x.openai_response)
        doc.set_ents(spans)
        output.append(doc)
    return output


def evaluate(*, gold: List[Doc], pred: List[Doc]) -> Dict[str, float]:
    examples = [Example(p, g) for g, p in zip(gold, pred)]
    scorer = Scorer()
    scores = scorer.score(examples)
    return scores


def _get_openai_spans(doc: Doc, response: str) -> List[Span]:
    spacy_spans = []
    for label, phrases in _parse_response(response):
        label = _normalize_label(label)
        offsets = _find_substrings(doc.text, phrases)
        for start, end in offsets:
            span = doc.char_span(start, end, alignment_mode="contract", label=label)
            if span is not None:
                spacy_spans.append(span)
        # This step prevents the same token from being used in multiple spans.
        # If there's a conflict, the longer span is preserved.
        spacy_spans = filter_spans(spacy_spans)
    return spacy_spans


def _parse_response(text: str) -> List[Tuple[str, List[str]]]:
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
            label = _normalize_label(label)
            if phrases.strip():
                phrases = [phrase.strip() for phrase in phrases.strip().split(",")]
                output.append((label, phrases))
    return output


def _normalize_label(label: str) -> str:
    return label.lower()


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


def main(spacy_lang: str, spacy_eval_path: Path, prodigy_jsonl_path: Path):
    nlp = spacy.blank(spacy_lang)
    gold_docs = list(DocBin().from_disk(spacy_eval_path).get_docs(nlp.vocab))
    prodigy_dataset = [
        ProdigyNER.from_dict(eg) for eg in srsly.read_jsonl(prodigy_jsonl_path)
    ]
    prodigy_dataset = find_to_evaluate(gold_docs, prodigy_dataset)
    pred_docs = make_docs(nlp, prodigy_dataset)
    assert len(gold_docs) == len(pred_docs)
    scores = evaluate(gold=gold_docs, pred=pred_docs)
    print(srsly.json_dumps(scores, indent=2))


if __name__ == "__main__":
    typer.run(main)
