import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
import spacy

from recipes.openai_ner import PromptExample, _find_substrings, _load_template

from .utils import make_suggester


def test_multiple_substrings():
    text = "The Blargs is the debut album by rock band The Blargs."
    substrings = ["The Blargs", "rock"]
    res = _find_substrings(text, substrings, single_match=False)
    assert res == [(0, 10), (43, 53), (33, 37)]
    res = _find_substrings(text, substrings, single_match=True)
    assert res == [(0, 10), (33, 37)]


def test_substrings_case():
    text = "A, a, B, b, a,b,c,d"
    substrings = ["a,"]
    res = _find_substrings(text, substrings, single_match=False, case_sensitive=True)
    assert res == [(3, 5), (12, 14)]
    res = _find_substrings(text, substrings, single_match=False, case_sensitive=False)
    assert res == [(0, 2), (3, 5), (12, 14)]
    res = _find_substrings(text, substrings, single_match=True, case_sensitive=True)
    assert res == [(3, 5)]
    res = _find_substrings(text, substrings, single_match=True, case_sensitive=False)
    assert res == [(0, 2)]


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
PERSON: <JSON list of strings>
PLACE: <JSON list of strings>
PERIOD: <JSON list of strings>

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
        PromptExample(
            text="New York is a large city.", entities={"PLACE": ["New York"]}
        ),
        PromptExample(
            text="David Hasslehoff and Helena Fischer are big in Germany.",
            entities={
                "PERSON": ["David Hasslehoff", "Helena Fischer"],
                "PLACE": ["Germany"],
            },
        ),
    ]
    path = Path(__file__).parent.parent / "templates" / "ner_prompt.jinja2"
    template = _load_template(path)
    prompt = template.render(text=text, labels=labels, examples=examples)
    assert (
        prompt
        == f"""
From the text below, extract the following entities in the following format:
PERSON: <JSON list of strings>
PLACE: <JSON list of strings>
PERIOD: <JSON list of strings>

Text:
\"\"\"
David Bowie lived in Berlin in the 1970s.
\"\"\"

For example:

Text:
\"\"\"
New York is a large city.
\"\"\"
PLACE: ["New York"]

Text:
\"\"\"
David Hasslehoff and Helena Fischer are big in Germany.
\"\"\"
PERSON: ["David Hasslehoff", "Helena Fischer"]
PLACE: ["Germany"]

""".lstrip()
    )


@pytest.mark.parametrize(
    "comment,text,raw_spans,filtered_spans",
    [
        (
            "Does it take longest when nested and shorter is first?",
            "one two three four",
            [("a", 0, 1), ("b", 0, 2)],
            [("b", 0, 2)],
        ),
        (
            "Does it take longest when nested and shorter is last?",
            "one two three four",
            [("b", 0, 2), ("a", 0, 1)],
            [("b", 0, 2)],
        ),
        (
            "Does it take first when overlapping and shorter is first?",
            "one two three four",
            [("b", 0, 2), ("a", 1, 3)],
            [("b", 0, 2)],
        ),
        (
            "Does it take first when overlapping and shorter is last?",
            "one two three four",
            [("a", 0, 2), ("b", 1, 3)],
            [("a", 0, 2)],
        ),
    ],
)
def test_one_token_per_span(
    comment: str,
    text: str,
    raw_spans: List[Tuple[str, int, int]],
    filtered_spans: List[Tuple[int, int]],
):
    labels = list(sorted(set(label for label, _, _1 in raw_spans)))
    suggester = make_suggester(
        labels=labels, openai_api_key="fake api key", openai_api_org="fake api org"
    )
    prompt = suggester._get_ner_prompt(text, labels=labels, examples=[])
    response = _get_response(text, labels, raw_spans)
    stream = [{"text": text, "openai": {"prompt": prompt, "response": response}}]
    stream = list(suggester.format_suggestions(stream, nlp=spacy.blank("en")))
    output_spans = [
        (s["label"], s["token_start"], s["token_end"] + 1) for s in stream[0]["spans"]
    ]
    assert output_spans == filtered_spans
    # Also check there's no overlaps
    seen_tokens = set()
    for _, start, end in output_spans:
        for i in range(start, end):
            assert i not in seen_tokens, "Overlapping or nested spans found"
            seen_tokens.add(i)


def _get_response(text: str, labels, spans: List[Tuple[str, int, int]]) -> str:
    # Get table of start and end character offsets for the test spans.
    tokens = text.split()
    start_chars, end_chars = _get_token_char_maps(tokens, [True for _ in tokens])
    spans_by_label = defaultdict(list)
    for label, start, end in spans:
        spans_by_label[label].append(text[start_chars[start] : end_chars[end - 1]])
    response_lines = []
    for label in labels:
        response_lines.append(f"{label}: {json.dumps(spans_by_label[label])}")
    return "\n".join(response_lines)


def _get_token_char_maps(
    tokens: List[str], whitespace: List[bool]
) -> Tuple[Dict[int, int], Dict[int, int]]:
    idx = 0
    start_chars = {}
    end_chars = {}
    for i, token in enumerate(tokens):
        start_chars[i] = idx
        idx += len(token) + int(whitespace[i])
        end_chars[i] = idx
    return start_chars, end_chars
