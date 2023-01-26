"""A/B evaluation of OpenAI responses, for prompt engineering.

A/B evaluation is basically a blind "taste test". Results are produced
from two experimental conditions (in this case, responses from two
prompts). The annotator is shown the response pair, without knowing which
condition produced which response. The annotator marks which one is better,
and the response is recorded.

At the end of annotation, the results are tallied up, so you can see whether
one condition produces better results than the other. This lets you apply
a sound methodology to subjective decisions.
"""
from collections import Counter
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar, cast

import httpx
import jinja2

import prodigy
import pydantic
import srsly
from dotenv import load_dotenv
from prodigy.util import msg

_ItemT = TypeVar("_ItemT")
# Set up openai
load_dotenv()  # take environment variables from .env.


class PromptInput(pydantic.BaseModel):
    id: str
    prompt_args: Dict[str, Any]


class OpenAIPromptAB:
    display: jinja2.Template
    prompts: Dict[str, jinja2.Template]
    inputs: Iterable[PromptInput]
    batch_size: int
    verbose: bool
    randomize: bool
    openai_api_org: str
    openai_api_key: str
    openai_temperature: float
    openai_max_tokens: int
    openai_timeout_s: int
    openai_n: int

    def __init__(
        self,
        display: jinja2.Template,
        prompts: Dict[str, jinja2.Template],
        inputs: Iterable[PromptInput],
        *,
        openai_api_org: str,
        openai_api_key: str,
        openai_model: str,
        batch_size: int = 10,
        verbose: bool = False,
        randomize: bool = True,
        openai_temperature: float = 0,
        openai_max_tokens: int = 500,
        openai_timeout_s: int = 50,
        openai_n: int = 1,
        repeat: int = 3
    ):
        self.display = display
        self.inputs = inputs
        self.prompts = prompts
        self.model = openai_model
        self.batch_size = batch_size
        self.verbose = verbose
        self.openai_api_org = openai_api_org
        self.openai_api_key = openai_api_key
        self.openai_temperature = openai_temperature
        self.openai_max_tokens = openai_max_tokens
        self.openai_timeout_s = openai_timeout_s
        self.openai_n = openai_n
        self.randomize = randomize
        self.repeat = repeat

    def __iter__(self) -> Iterable[Dict]:
        for input_batch in _batch_sequence(self.inputs, self.batch_size):
            for _ in range(self.repeat):
                response_batch = self._get_response_batch(input_batch)
                for input_, responses in zip(input_batch, response_batch):
                    yield self._make_example(
                        input_.id,
                        self.display.render(**input_.prompt_args),
                        responses,
                        randomize=self.randomize,
                    )

    def on_exit(self, ctrl):
        examples = ctrl.db.get_dataset_examples(ctrl.dataset)
        counts = Counter()
        # Get last example per ID
        for eg in examples:
            selected = eg.get("accept", [])
            if not selected or len(selected) != 1 or eg["answer"] != "accept":
                continue
            counts[selected[0]] += 1
        print("")
        if not counts:
            msg.warn("No answers found", exits=0)
        msg.divider("Evaluation results", icon="emoji")
        pref, _ = counts.most_common(1)[0]
        msg.good(f"You preferred {pref}")
        rows = [(name, count) for name, count in counts.most_common()]
        msg.table(rows, aligns=("l", "r"))

    def _get_response_batch(self, inputs: List[PromptInput]) -> List[Dict[str, str]]:
        name1, name2 = self._choose_rivals()
        prompts = []
        for input_ in inputs:
            prompts.append(self._get_prompt(name1, input_.prompt_args))
            prompts.append(self._get_prompt(name2, input_.prompt_args))
        responses = self._get_responses(prompts)
        assert len(responses) == len(inputs) * 2
        output = []
        # Pair out the responses. There's a fancy
        # zip way to do this but I think that's less
        # readable
        for i in range(0, len(responses), 2):
            output.append({name1: responses[i], name2: responses[i + 1]})
        return output

    def _choose_rivals(self) -> Tuple[str, str]:
        assert len(self.prompts) == 2
        return tuple(sorted(self.prompts.keys()))

    def _get_prompt(self, name: str, args: Dict) -> str:
        return self.prompts[name].render(**args)

    def _get_responses(self, prompts: List[str]) -> List[str]:
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "OpenAI-Organization": self.openai_api_org,
            "Content-Type": "application/json",
        }
        r = _retry429(
            lambda: httpx.post(
                "https://api.openai.com/v1/completions",
                headers=headers,
                json={
                    "model": self.model,
                    "prompt": prompts,
                    "temperature": self.openai_temperature,
                    "max_tokens": self.openai_max_tokens,
                },
                timeout=self.openai_timeout_s,
            ),
            n=self.openai_n,
            timeout_s=self.openai_timeout_s,
        )
        r.raise_for_status()
        responses = r.json()
        return [responses["choices"][i]["text"].strip() for i in range(len(prompts))]

    def _make_example(
        self, id: str, input: str, responses: Dict[str, str], randomize: bool
    ) -> Dict:

        question = {
            "id": id,
            "text": input,
            "options": [],
        }
        response_pairs = list(responses.items())
        if randomize:
            random.shuffle(response_pairs)
        else:
            response_pairs = list(sorted(response_pairs))
        for name, value in response_pairs:
            question["options"].append({"id": name, "text": value})
        return question


@prodigy.recipe(
    # fmt: off
    "ab.openai.prompts",
    dataset=("Dataset to save answers to", "positional", None, str),
    inputs_path=("Path to jsonl inputs", "positional", None, Path),
    display_template_path=("Template for summarizing the arguments","positional", None, Path),
    prompt1_template_path=("The first jinja2 prompt template","positional", None, Path),
    prompt2_template_path=("Path to second jinja2 prompt template","positional", None, Path),
    model=("GPT-3 model to use for responses", "option", "m", str),
    batch_size=("Batch size to send to OpenAI API", "option", "b", int),
    verbose=("Print extra information to terminal", "flag", "v", bool),
    no_random=("Don't randomize which annotation is shown as correct","flag","NR",bool,),
    repeat=("How often to send the same prompt to OpenAI", "option", "r", int)
    # fmt: on
)
def ab_openai_prompts(
    dataset: str,
    inputs_path: Path,
    display_template_path: Path,
    prompt1_template_path: Path,
    prompt2_template_path: Path,
    model: str = "text-davinci-003",
    batch_size: int = 10,
    verbose: bool = False,
    no_random: bool = False,
    repeat: int = 1,
):
    api_key, api_org = _get_api_credentials(model)
    inputs = [PromptInput(**x) for x in cast(List[Dict], srsly.read_jsonl(inputs_path))]

    display = _load_template(display_template_path)
    prompt1 = _load_template(prompt1_template_path)
    prompt2 = _load_template(prompt2_template_path)
    stream = OpenAIPromptAB(
        display=display,
        prompts={
            prompt1_template_path.name: prompt1,
            prompt2_template_path.name: prompt2,
        },
        inputs=inputs,
        openai_api_org=api_org,
        openai_api_key=api_key,
        openai_model=model,
        batch_size=batch_size,
        verbose=verbose,
        randomize=not no_random,
        openai_temperature=0.9,
        repeat=repeat
    )
    return {
        "dataset": dataset,
        "view_id": "choice",
        "stream": stream,
        "on_exit": stream.on_exit,
        "config": {
            "batch_size": batch_size,
            "exclude_by": "input",
            "global_css": ".prodigy-content{line-height: 1.2;};"
        },
    }

def _get_api_credentials(model: str) -> Tuple[str, str]:
    # Fetch and check the key
    api_key = os.getenv("OPENAI_KEY")
    if api_key is None:
        m = (
            "Could not find the API key to access the openai API. Ensure you have an API key "
            "set up via https://beta.openai.com/account/api-keys, then make it available as "
            "an environment variable 'OPENAI_KEY', for instance in a .env file."
        )
        msg.fail(m)
        sys.exit(-1)
    # Fetch and check the org
    org = os.getenv("OPENAI_ORG")
    if org is None:
        m = (
            "Could not find the organisation to access the openai API. Ensure you have an API key "
            "set up via https://beta.openai.com/account/api-keys, obtain its organization ID 'org-XXX' "
            "via https://beta.openai.com/account/org-settings, then make it available as "
            "an environment variable 'OPENAI_ORG', for instance in a .env file."
        )
        msg.fail(m)
        sys.exit(-1)

    # Check the access and get a list of available models to verify the model argument (if not None)
    # Even if the model is None, this call is used as a healthcheck to verify access.
    headers = {
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Organization": org,
    }
    r = _retry429(
        lambda: httpx.get(
            "https://api.openai.com/v1/models",
            headers=headers,
        ),
        n=1,
        timeout_s=1,
    )
    if r.status_code == 422:
        m = (
            "Could not access api.openai.com -- 422 permission denied."
            "Visit https://beta.openai.com/account/api-keys to check your API keys."
        )
        msg.fail(m)
        sys.exit(-1)
    elif r.status_code != 200:
        m = "Error accessing api.openai.com" f"{r.status_code}: {r.text}"
        msg.fail(m)
        sys.exit(-1)

    if model is not None:
        response = r.json()["data"]
        models = [response[i]["id"] for i in range(len(response))]
        if model not in models:
            e = f"The specified model '{model}' is not available. Choices are: {sorted(set(models))}"
            msg.fail(e, exits=1)

    return api_key, org


def _load_template(path: Path) -> jinja2.Template:
    # I know jinja has a lot of complex file loading stuff,
    # but we're not using the inheritance etc that makes
    # that stuff worthwhile.
    if not path.suffix == ".jinja2":
        msg.fail(
            "The parameter expects a .jinja2 file.",
            exits=1,
        )
    with path.open("r", encoding="utf8") as file_:
        text = file_.read()
    return jinja2.Template(text)


def _retry429(
    call_api: Callable[[], httpx.Response], n: int, timeout_s: int
) -> httpx.Response:
    """Retry a call to the OpenAI API if we get a 429: Too many requests
    error.
    """
    assert n >= 0
    assert timeout_s >= 1
    r = call_api()
    i = -1
    while i < n and r.status_code == 429:
        time.sleep(timeout_s)
        i += 1
    return r


def _batch_sequence(items: Iterable[_ItemT], batch_size: int) -> Iterable[List[_ItemT]]:
    batch = []
    for eg in items:
        batch.append(eg)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
