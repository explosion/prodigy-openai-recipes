import pytest

from recipes.openai_textcat import make_textcat_response_parser


@pytest.mark.parametrize(
    # fmt: off
    "response,answer",
    [
        ("Answer: accept\nReason: The text is a recipe.", "accept"),
        ("Answer: Accept\nReason: The text is a recipe.", "accept"),
        ("Answer: reject\nReason: The text is not a recipe.", "reject"),
        ("Answer: Reject\nReason: The text is not a recipe.", "reject"),
        ("answer: reject\nreason: The text is not a recipe.", "reject"),
        ("answer: Reject\nreason: The text is not a recipe.\nI don't know what it's about.", "reject"),
    ],
    # fmt: on
)
def test_parse_response_binary(response, answer):
    """Test if parse response works for common examples"""
    labels = ["recipe"]  # binary
    parser = make_textcat_response_parser(labels=labels)
    example = parser(response)
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
    parser = make_textcat_response_parser(labels=labels)
    example = parser(response)
    assert set(example.get("accept")) == set(answer)


@pytest.mark.parametrize("labels", [["binary"], ["multi1", "multi2"]])
def test_parser_no_answer(labels):
    """Test if parse response works for common examples"""
    empty_response = ""
    parser = make_textcat_response_parser(labels=labels)
    example = parser(empty_response)
    assert not example.get("accept")
    assert not example.get("reason")


@pytest.mark.parametrize("labels", [["binary"], ["multi1", "multi2"]])
@pytest.mark.parametrize(
    "response", ["asdfghjklmnop", "I am now a sentient robot. Bow before me."]
)
def test_parser_openai_returns_arbitrary_text(labels, response):
    """Test if parser response works for any arbitrary text"""
    parser = make_textcat_response_parser(labels=labels)
    example = parser(response)
    assert not example.get("accept")
    assert not example.get("reason")
