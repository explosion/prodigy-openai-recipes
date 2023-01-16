from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

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
single_completion = "monopoly"


test_cases = [
    (base_completion, ["monopoly", "Scrabble"]),
    (base_completion_with_trailing_spaces, ["monopoly", "scrabble"]),
    (trailing_token_completion, ["monopoly", "scrabble", "Risk"]),
    (single_completion, ["monopoly"]),
]


@pytest.mark.parametrize("completion,expectation", test_cases)
def test_parse_terms(completion, expectation):
    assert _parse_terms(completion=completion) == expectation
