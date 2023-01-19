import pytest
from typing import List
from pathlib import Path

import spacy

from recipes.openai_textcat import TextCatOpenAISuggester, DEFAULT_PROMPT_PATH
from recipes.openai import get_api_credentials, load_template, OpenAISuggester


def make_suggester(
    suggester: OpenAISuggester,
    labels: List[str],
    prompt_path: Path,
    model: str = "text-davinci-003",
    **kwargs
) -> OpenAISuggester:
    if "openai_api_key" not in kwargs or "openai_api_org" not in kwargs:
        api_key, api_org = get_api_credentials(model)
        if "openai_api_key" not in kwargs:
            kwargs["openai_api_key"] = api_key
        if "openai_api_org" not in kwargs:
            kwargs["openai_api_org"] = api_org
    if "max_examples" not in kwargs:
        kwargs["max_examples"] = 0
    if "prompt_template" not in kwargs:
        kwargs["prompt_template"] = load_template(prompt_path)
    if "segment" not in kwargs:
        kwargs["segment"] = False
    if "openai_model" not in kwargs:
        kwargs["openai_model"] = "text-davinci-003"
    return suggester(labels=labels, **kwargs)


@pytest.mark.parametrize(
    # fmt: off
    "response,answer",
    [
        ("Answer: accept\nReason: The text is a recipe.", True),
        ("Answer: Accept\nReason: The text is a recipe.", True),
        ("Answer: reject\nReason: The text is not a recipe.", False),
        ("Answer: Reject\nReason: The text is not a recipe.", False),
        ("answer: reject\nreason: The text is not a recipe.", False),
        ("answer: Reject\nreason: The text is not a recipe.\nI don't know what it's about.", False),
    ],
    # fmt: on
)
def test_parse_response_binary(response, answer):
    """Test if parse response works for common examples"""
    labels = ["recipe"]  # binary
    suggester = make_suggester(
        TextCatOpenAISuggester, prompt_path=DEFAULT_PROMPT_PATH, labels=labels
    )
    nlp = spacy.blank("en")
    example = suggester.parse_response(example={}, response=response, nlp=nlp)
    assert example.get("answer") == answer


@pytest.mark.parametrize(
    # fmt: off
    "response,answer",
    [
        ("Answer: recipe,feedback,question\nReason: It is all three!", ["recipe", "feedback", "question"]),
        ("Answer: recipe\nReason: The text is a recipe.", ["recipe"]),
        ("Answer: recipe,\nReason: The text is a recipe.", ["recipe"]),
        ("Answer: recipe,feedback,\nReason: The text is a feedback about a recipe.\nThat's what I think...", ["recipe", "feedback"]),
        ("answer: recipe,feedback,\nreason: The text is a feedback about a recipe.\nThat's what I think...", ["recipe", "feedback"]),
        ("answer: \nreason: It's none of the above.", []),
    ],
    # fmt: on
)
def test_parser_response_multi(response, answer):
    """Test if parse response works for common examples"""
    labels = ["recipe", "feedback", "question"]  # multiclass
    suggester = make_suggester(
        TextCatOpenAISuggester, prompt_path=DEFAULT_PROMPT_PATH, labels=labels
    )
    nlp = spacy.blank("en")
    example = suggester.parse_response(example={}, response=response, nlp=nlp)
    assert set(example.get("accept")) == set(answer)


@pytest.mark.parametrize("labels", [["binary"], ["multi1", "multi2"]])
def test_parser_no_answer(labels):
    """Test if parse response works for common examples"""
    suggester = make_suggester(
        TextCatOpenAISuggester, prompt_path=DEFAULT_PROMPT_PATH, labels=labels
    )
    nlp = spacy.blank("en")
    example = suggester.parse_response(example={}, response="", nlp=nlp)
    assert not example.get("accept")
    assert not example.get("reason")


@pytest.mark.parametrize("labels", [["binary"], ["multi1", "multi2"]])
@pytest.mark.parametrize(
    "response", ["asdfghjklmnop", "I am now a sentient robot. Bow before me."]
)
def test_parser_openai_returns_arbitrary_text(labels, response):
    """Test if parser response works for any arbitrary text"""
    suggester = make_suggester(
        TextCatOpenAISuggester, prompt_path=DEFAULT_PROMPT_PATH, labels=labels
    )
    nlp = spacy.blank("en")
    example = suggester.parse_response(example={}, response=response, nlp=nlp)
    assert not example.get("accept")
    assert not example.get("reason")
