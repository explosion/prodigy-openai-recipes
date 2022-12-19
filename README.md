# Prodigy OpenAI recipes

This repository contains example code on how to combine **zero- and few-shot learning
with a small annotation effort** to obtain a **high-quality dataset with maximum efficiency**. Specifically, we use large language models available from [OpenAI](https://openai.com) to provide us with an initial set of predictions,
then spin up a [Prodigy](https://prodi.gy) instance on our local machine
to go through these predictions and curate them. This allows us to obtain a
gold-standard dataset pretty quickly, and train a smaller, supervised model that fits
our exact needs and use-case.

![](https://user-images.githubusercontent.com/13643239/208497043-178beb47-f7c6-4b3e-a253-9e12e2f0c807.png)

https://user-images.githubusercontent.com/13643239/208504034-0ab6bcbe-6d2b-415d-8257-233f2074ba31.mp4

## Setup and Install

Make sure to [install Prodigy](https://prodi.gy/docs/install) as well as a few additional Python dependencies:

```bash
python -m pip install prodigy -f https://XXXX-XXXX-XXXX-XXXX@download.prodi.gy
python -m pip install -r requirements.txt
```

With `XXXX-XXXX-XXXX-XXXX` being your personal Prodigy license key.

Then, create a new API key from [openai.com](https://beta.openai.com/account/api-keys) or fetch an existing
one. Record the secret key as well as the [organization key](https://beta.openai.com/account/org-settings)
and make sure these are available as environmental variables. For instance, set them in a `.env` file in the
root directory:

```
OPENAI_ORG = "org-..."
OPENAI_KEY = "sk-..."
```

## `ner.openai.correct`: NER annotation with zero- or few-shot learning

This recipe marks entity predictions obtained from a large language model and allows you to flag them as correct, or to
manually curate them. This allows you to quickly gather a gold-standard dataset through zero-shot or few-shot learning.
It's very much like using the standard [`ner.correct`](https://prodi.gy/docs/recipes#ner-correct) recipe in Prodi.gy,
but we're using GPT-3 as a backend model to make predictions.

```bash
python -m prodigy ner.openai.correct dataset filepath labels [--options] -F ./recipes/openai_ner.py
```

| Argument                | Type | Description                                                                                                                                     | Default                         |
| ----------------------- | ---- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------- |
| `dataset`               | str  | Prodigy dataset to save annotations to.                                                                                                         |                                 |
| `file_path`             | Path | Path to `.jsonl` data to annotate. The data should at least contain a `"text"` field.                                                           |                                 |
| `labels`                | str  | Comma-separated list defining the NER labels the model should predict.                                                                          |                                 |
| `--lang`, `-l`          | str  | Language of the input data - will be used to obtain a relevant tokenizer.                                                                       | `"en"`                          |
| `--segment`, `-n`       | bool | Flag to set when examples should be split into sentences. By default, the full input article is shown.                                          | `False`                         |
| `--model`, `-m`         | str  | GPT-3 model to use for initial predictions.                                                                                                     | `"text-davinci-003"`            |
| `--prompt_path`, `-p`   | Path | Path to the `.jinja2` [prompt template](templates).                                                                                             | `./templates/ner_prompt.jinja2` |
| `--examples-path`, `-e` | str  | Path to examples to help define the task. The file can be a .yml, .yaml or .json. If set to `None`, zero-shot learning is applied.              | `None`                          |
| `--max-examples`, `-n`  | int  | Max number of examples to include in the prompt to OpenAI. If set to 0, zero-shot learning is always applied, even when examples are available. | 2                               |
| `--batch-size`, `-b`    | int  | Batch size of queries to send to the OpenAI API.                                                                                                | 10                              |
| `--verbose`, `-v`       | bool | Flag to print extra information to the terminal.                                                                                                | `False`                         |

### Example usage

Let's say we want to recognize dishes, ingredients and cooking equipment from some text we obtained from a cooking subreddit.
We'll send the text to GPT-3, hosted by OpenAI, and provide an annotation prompt to explain
to the language model the type of predictions we want. Something like:

```
From the text below, extract the following entities in the following format:
dish: <comma delimited list of strings>
ingredient: <comma delimited list of strings>
equipment: <comma delimited list of strings>

Text:
...
```

We define the definition of this prompt in a .jinja2 file which also describes how to append examples for few-shot learning.
You can create your own [template](templates) and provide it to the recipe with the `--prompt-path` or `-p` option.
Additionally, with `--examples-path` or `-e` you can set the file path of a .y(a)ml or .json file that contains additional examples:

```bash
python -m prodigy ner.openai.correct my_ner_data ./data/reddit_r_cooking_sample.jsonl "dish,ingredient,equipment"
-p ./templates/ner_prompt.jinja2 -e ./examples/input.yaml -n 2 -F ./recipes/openai_ner.py
```

After receiving the results from the OpenAI API, the Prodigy recipe converts the predictions into an annotation task
that can be rendered with Prodigy. The task even shows the original prompt as well as the raw answer we obtained
from the language model.

<img src="https://user-images.githubusercontent.com/8796347/208484904-72fd79e4-9f14-4c40-9993-97a5776aafb3.png" width="600" />

Here, we see that the model is able to correctly recognize dishes, ingredients and cooking equipment right from the start!

The recipe also offers a `--verbose` or `-v` option that includes the exact prompt and response on the terminal as traffic is received.
Note that because the requests to the API are batched, you might have to scroll back a bit to find the current prompt.

### Interactively tune the prompt examples

At some point, you might notice a mistake in the predictions of the OpenAI language model. For instance, we noticed an error
in the recognition of cooking equipment in this example:

<img src="https://user-images.githubusercontent.com/8796347/208485149-a32fa2da-db8a-42a5-a2a7-4708f127b592.png" width="600" />

If you see these kind of systematic errors, you can steer the predictions in the right direction by correcting the example and then selecting the small "flag" icon
in the top right of the Prodigy UI:

<img src="https://user-images.githubusercontent.com/8796347/208485453-861fdecf-6283-4a8f-b802-58314f3e496d.png" width="600" />

Once you hit <kbd>accept</kbd> on the Prodigy interface, the flagged example will be automatically picked up and added to the examples
that are sent to the OpenAI API as part of the prompt. Note that because Prodigy batches these requests,
the prompt will be updated with a slight delay, after the next batch of prompts is sent to OpenAI.
You can experiment with making the batch size (`--batch-size` or `-b`) smaller to have the change come into effect sooner,
but this might negatively impact the speed of the annotation workflow.

## `ner.openai.fetch`: Fetch examples up-front

The `ner.openai.correct` recipe fetches examples from OpenAI while annotating, but we've also included a recipe that can fetch a large batch of examples upfront.

```bash
python -m prodigy ner.openai.fetch input_data.jsonl predictions.jsonl "dish,ingredient,equipment" -F ./recipes/ner.py
```

This will create a `predictions.jsonl` file that can be loaded with the [`ner.manual`](https://prodi.gy/docs/recipes#ner-manual) recipe.

Note that the OpenAI API might return "429 Too Many Request" errors when requesting too much data at once - in this case it's best to ensure you only request
100 or so examples at a time.

## Exporting the annotations and training an NER model

After you've curated a set of predictions, you can export the results with [`db-out`](https://prodi.gy/docs/recipes#db-out):

```bash
python -m prodigy db-out my_ner_data  > ner_data.jsonl
```

The format of the exported annotations contains all the data you need to train a smaller model downstream. Each example
in the dataset contains the original text, the tokens, span annotations denoting the entities, etc.

You can also export the data to spaCy's [binary format](https://spacy.io/api/data-formats#training), using [`data-to-spacy`](https://prodi.gy/docs/recipes#data-to-spacy). This format lets you load in the annotations as spaCy `Doc` objects, which can be convenient for further conversion. The `data-to-spacy` command also makes it easy to train an NER model with spaCy, as follows:

```bash
python -m prodigy data-to-spacy ./data/annotations3/ --ner mydata -es 0.2
python -m spacy train ./data/annotations3/config.cfg --paths.train ./data/annotations3/train.spacy --paths.dev ./data/annotations3/dev.spacy
```

We've also included an experimental script to load in the `.spacy` binary format and train a model with the HuggingFace `transformers` library. You can convert export the annotations and run the script like this:

```
pip install -r requirements_train.txt
python -m prodigy data-to-spacy ./data/annotations2 --ner my_ner_data -es 0
python scripts/train_hf_ner.py ./data/annotations2/train.spacy ner-model
```

This will create the file `./data/annotations2/train.spacy` with your annotated documents (you'll also see a warning about not creating evaluation data, but that's OK, the training script will create it). The resulting model will be saved to the `ner-model/` directory.
