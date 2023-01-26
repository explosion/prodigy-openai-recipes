import pytest

from recipes.openai_terms import _parse_terms

# We cannot assume that `risk` is a complete game because
# OpenAI may have exhausted the tokens
base_completion = """monopoly
- Scrabble
- risk
"""

# Added some trailing spaces in this one
base_completion_with_trailing_spaces = """monopoly 
- scrabble 
- risk 
"""

# An example where the tokens may have been exhausted
# note the capitalisation
trailing_token_completion = """monopoly
- scrabble
- Risk
- carcas
"""

# This can also happen
single_line_completion = "monopoly"


@pytest.mark.parametrize(
    "comment,completion,expectation",
    [
        (
            "Base OpenAI completion with capitalisation",
            base_completion,
            ["monopoly", "Scrabble"],
        ),
        (
            "Check trailing spaces",
            base_completion_with_trailing_spaces,
            ["monopoly", "scrabble"],
        ),
        (
            "Completion with bad final item",
            trailing_token_completion,
            ["monopoly", "scrabble", "Risk"],
        ),
        (
            "Example of a single-line OpenAI completion",
            single_line_completion,
            ["monopoly"],
        ),
    ],
)
def test_parse_terms(comment, completion, expectation):
    assert _parse_terms(completion=completion) == expectation
