from typing import List

from recipes.openai_ner import (DEFAULT_PROMPT_PATH, OpenAISuggester,
                                _get_api_credentials, _load_template)


def make_suggester(labels: List[str], **kwargs) -> OpenAISuggester:
    if "openai_api_key" not in kwargs or "openai_api_org" not in kwargs:
        api_key, api_org = _get_api_credentials(model)
        if "openai_api_key" not in kwargs:
            kwargs["openai_api_key"] = api_key
        if "openai_api_org" not in kwargs:
            kwargs["openai_api_org"] = api_org
    if "max_examples" not in kwargs:
        kwargs["max_examples"] = 0
    if "prompt_template" not in kwargs:
        kwargs["prompt_template"] = _load_template(DEFAULT_PROMPT_PATH)
    if "segment" not in kwargs:
        kwargs["segment"] = False
    if "openai_model" not in kwargs:
        kwargs["openai_model"] = "text-davinci-003"
    return OpenAISuggester(labels=labels, **kwargs)
