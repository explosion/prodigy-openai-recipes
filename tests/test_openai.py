from typing import List
import pytest
import jinja2
import httpx
from pytest_httpx import HTTPXMock

from recipes.openai import OpenAISuggester, get_api_credentials


# Setup the template and the suggester
environment = jinja2.Environment()
template = environment.from_string("Prompt: {{ text }}")

openai = OpenAISuggester(
    prompt_template=template,
    labels=["label1", "label2"],
    max_examples=1,
    segment=False,
    openai_model="text-davinci-003",
    openai_api_key="Fake api key",
    openai_api_org="Fake api org",
    response_parser=lambda x: {"key": "value"},
    openai_n_retries=1,
    openai_read_timeout_s=1,
    openai_retry_timeout_s=1,
    prompt_example_class=None,
)


@pytest.mark.parametrize(
    "prompts,response_text",
    [
        (["A single prompt"], ["A single response"]),
        (["A batch", "of prompts"], ["A batch", "of responses"]),
    ],
)
def test_openai_response_follows_contract(
    httpx_mock: HTTPXMock, prompts: List[str], response_text
):
    """Test happy path where OpenAI follows the contract and we can parse it
    https://beta.openai.com/docs/api-reference/completions
    """

    httpx_mock.add_response(
        method="POST",
        json={
            "choices": [
                {
                    "text": text,
                    "index": index,
                    "logprobs": 0.1,
                    "finish_reason": "length",
                }
                for index, text in enumerate(response_text)
            ]
        },
    )

    chatgpt_response = openai._get_openai_response(prompts=prompts)
    assert len(chatgpt_response) == len(prompts)
    assert set(chatgpt_response) == set(response_text)


@pytest.mark.parametrize("error_code", openai.RETRY_ERROR_CODES)
def test_retry_function_when_calls_fail(httpx_mock, error_code):
    """Test if status error shows up after all failed retries."""
    httpx_mock.add_response(status_code=error_code)
    with pytest.raises(httpx.HTTPStatusError):
        openai._get_openai_response(prompts=["Some prompt"])


@pytest.mark.parametrize(
    "key,org", [(None, "fake api org"), ("fake api key", None), (None, None)]
)
def test_get_api_credentials_error_handling_envvars(monkeypatch, key, org):
    """Ensure that auth fails whenever key or org is none."""
    monkeypatch.setenv("OPENAI_KEY", key)
    monkeypatch.setenv("OPENAI_ORG", org)
    with pytest.raises(SystemExit):
        get_api_credentials(model="text-davinci-003")


@pytest.mark.parametrize("error_code", [422, 500, 501])
def test_get_api_credentials_calls_fail(httpx_mock, monkeypatch, error_code):
    """Ensure that auth fails when we encounter an error code."""
    httpx_mock.add_response(status_code=error_code)
    monkeypatch.setenv("OPENAI_KEY", "fake api key")
    monkeypatch.setenv("OPENAI_ORG", "fake api org")

    with pytest.raises(SystemExit):
        get_api_credentials(model="text-davinci-003")


def test_get_api_credentials_model_does_not_exist(httpx_mock, monkeypatch):
    """Ensure that auth fails when model passed does not exist.
    https://beta.openai.com/docs/api-reference/models/list
    """
    httpx_mock.add_response(
        method="GET",
        json={
            "data": [{"id": "model-id-0"}, {"id": "model-id-1"}, {"id": "model-id-2"}]
        },
    )
    monkeypatch.setenv("OPENAI_KEY", "fake api key")
    monkeypatch.setenv("OPENAI_ORG", "fake api org")

    with pytest.raises(SystemExit):
        get_api_credentials(model="a-model-that-does-not-exist")
