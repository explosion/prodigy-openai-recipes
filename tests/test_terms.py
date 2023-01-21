import pytest

from recipes.openai_terms import _parse_terms, _parse_variants

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


# I've seen variants with numbers appear despite the prompt
completion_with_numbers = """1. 5050 slide 
2. Fifty-Fifty slide 
3. 5-0 slide 
4. Five-Oh slide 
5. 5050 smith grind
"""

@pytest.mark.parametrize(
    "comment,completion,expectation",
    [
        (
            "Base OpenAI completion with capitalisation",
            base_completion,
            ["monopoly", "Scrabble", "risk"],
        ),
        (
            "Check trailing spaces",
            base_completion_with_trailing_spaces,
            ["monopoly", "scrabble", "risk"],
        ),
        (
            "Completion with bad final item",
            trailing_token_completion,
            ["monopoly", "scrabble", "Risk", "carcas"],
        ),
        (
            "Example of a single-line OpenAI completion",
            single_line_completion,
            ["monopoly"],
        ),
        (
            "Example of numbered list",
            completion_with_numbers,
            ['5050 slide','FiftyFifty slide','50 slide', 'FiveOh slide', '5050 smith grind']
        )
    ],
)
def test_parse_variants(comment, completion, expectation):
    # We're re-using completions that `_parse_text` uses but notice
    # that because the variants recipe doesn't have to worry about
    # token limits, which means we always can assume the final term
    # is complete.
    assert _parse_variants(completion=completion) == expectation