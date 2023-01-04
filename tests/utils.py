from typing import List, Callable
from pathlib import Path

from recipes.openai import get_api_credentials, load_template, OpenAISuggester


def make_suggester(
    response_parser: Callable,
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
    return OpenAISuggester(response_parser=response_parser, labels=labels, **kwargs)
