import pytest

from recipes.openai_paraphrase import _parse_terms

# We cannot assume that the final line contains a complete utterance
base_completion = """ This is a sentence we want.
- So is this
- But this isn't because this sentence can be imcomple-
"""

@pytest.mark.parametrize(
    "comment,completion,expectation",
    [
        (
            "Base OpenAI completion",
            base_completion,
            ["This is a sentence we want.", "So is this"],
        )
    ],
)
def test_parse_terms(comment, completion, expectation):
    assert _parse_terms(completion=completion) == expectation
