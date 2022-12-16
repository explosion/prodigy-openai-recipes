from pathlib import Path
from recipes.ner import _find_substrings, _load_template


def test_multiple_substrings():
    text = "The Blargs is the debut album by rock band The Blargs."
    substrings = ["The Blargs", "rock"]
    res = _find_substrings(text, substrings, single_match=False)
    assert len(res) == 3, res
    res = _find_substrings(text, substrings, single_match=True)
    assert len(res) == 2, res


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
