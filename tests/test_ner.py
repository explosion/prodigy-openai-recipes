from pathlib import Path
from recipes.ner import _find_substrings, _load_template


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
