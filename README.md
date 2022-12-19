# Prodigy OpenAI recipes

This repository contains example code on how to combine zero- and few-shot learning
with a small annotation effort to obtain a high-quality dataset with maximum efficiency. Specifically, we use large language models available from OpenAI to provide us with an initial set of predictions,
then spin up a Prodigy instance on our local machine
to go through these predictions and curate them. This allows us to obtain a
gold-standard dataset pretty quickly, and train a smaller, supervised model that fits
our exact needs and use-case.

![](https://user-images.githubusercontent.com/13643239/208323588-ba039581-e924-461e-a389-7ba3d4244f5d.png)

## Setup and Install

Make sure to [install Prodigy](https://prodi.gy/docs/install) as well as a few additional Python dependencies:

```
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

## ner.openai.correct: NER annotation with zero- or few-shot learning

This recipe marks entity predictions obtained from a large language model and allows you to flag them as correct, or to
manually curate them. This allows you to quickly gather a gold-standard dataset through zero-shot or few-shot learning.
It's very much like using the standard [`ner.correct`](https://prodi.gy/docs/recipes#ner-correct) recipe in Prodi.gy,
but we're using GPT-3 as a backend model to make predictions.

```
python -m prodigy ner.openai.correct dataset filepath labels [--options] -F ./recipes/openai_ner.py
```

| Argument                | Type | Description                                                                                                                                     | Default                         |
| ----------------------- | ---- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------- |
| `dataset`               | str  | Prodigy dataset to save annotations to.                                                                                                         |                                 |
| `file_path`             | Path | Path to jsonl data to annotate. The data should at least contain a 'text' field."                                                               |                                 |
| `labels`                | str  | Comma-separated list defining the NER labels the model should predict.                                                                          |                                 |
| `--lang`, `-l`          | str  | Language of the input data - will be used to obtain a relevant tokenizer.                                                                       | `"en"`                          |
| `--segment`, `-n`       | bool | Flag to set when examples should be split into sentences. By default, the full input article is shown.                                          | `False`                         |
| `--model`, `-m`         | str  | GPT-3 model to use for initial predictions.                                                                                                     | `"text-davinci-003"`            |
| `--prompt_path`, `-p`   | Path | Path to the .jinja2 prompt template.                                                                                                            | `./templates/ner_prompt.jinja2` |
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
You can create your own template and provide it to the recipe with the `--prompt-path` or `-p` option.
Additionally, with `--examples-path` or `-e` you can set the file path of a .y(a)ml or .json file that contains additional examples:

```
python -m prodigy ner.openai.correct my_ner_data ./data/reddit_r_cooking_sample.jsonl "dish,ingredient,equipment"
-p ./templates/ner_prompt.jinja2 -e ./examples/input.yaml -n 2 -F ./recipes/openai_ner.py
```

After receiving the results from the OpenAI API, the Prodigy recipe converts the predictions into an annotation task
that can be rendered with Prodigy. The task even shows the original prompt as well as the raw answer we obtained
from the language model.

![](https://user-images.githubusercontent.com/8796347/208484904-72fd79e4-9f14-4c40-9993-97a5776aafb3.png)

Here, we see that the model is able to correctly recognize dishes, ingredients and cooking equipment right from the start!

The recipe also offers a `--verbose` or `-v` option that includes the exact prompt and response on the terminal as traffic is received.
Note that because the requests to the API are batched, you might have to scroll back a bit to find the current prompt.

### Interactively tune the prompt examples

At some point, you might notice a mistake in the predictions of the OpenAI language model. For instance, we noticed an error
in the recognition of cooking equipment in this example:

![](https://user-images.githubusercontent.com/8796347/208485149-a32fa2da-db8a-42a5-a2a7-4708f127b592.png)

If you see these kind of systematic errors, you can steer the predictions in the right direction by correcting the example and then selecting the small "flag" icon
in the top right of the Prodigy UI:

![](https://user-images.githubusercontent.com/8796347/208485453-861fdecf-6283-4a8f-b802-58314f3e496d.png)

Once you hit "accept" on the Prodigy interface, the flagged example will be automatically picked up and added to the examples
that are sent to the OpenAI API as part of the prompt. Note that because Prodigy batches these requests,
the prompt will be updated with a slight delay, after the next batch of prompts is sent to OpenAI.
You can experiment with making the batch size (`--batch-size` or `-b`) smaller to have the change come into effect sooner,
but this might negatively impact the speed of the annotation workflow.

### db-out: obtain the curated examples

After you've curated a set of predictions, you can obtain the results with [`db-out`](https://prodi.gy/docs/recipes#db-out):

```
python -m prodigy db-out my_ner_data  > ner_data.jsonl
```

The format of the exported annotations contains all the data you need to train a smaller model downstream. Each example
in the dataset contains the original text, the tokens, span annotations denoting the entities, etc.

If you want to inspect the flagged instances, you could do:

```
python -m prodigy db-out my_ner_data | grep \"flagged\":true > ner_prompt_examples.jsonl
```

## ner.openai.fetch: Fetch examples up-front

The `ner.openai.correct` recipe fetches examples from OpenAI while annotating, but we've also included a recipe that can fetch a large batch of examples upfront.

```
python -m prodigy ner.openai.fetch input_data.jsonl predictions.jsonl "dish,ingredient,equipment" -F ./recipes/ner.py
```

This will create a `predictions.jsonl` file that can be loaded with the [ner.manual](https://prodi.gy/docs/recipes#ner-manual) recipe.

Note that the OpenAI API might return "429 Too Many Request" errors when requesting too much data at once - in this case it's best to ensure you only request 
100 orso examples at a time.

## Training an NER model with Hugging Face

After you've annotated enough examples - say 100 to start - you can try training a model. We've included a script to automatically train a model using HuggingFace's Transformers library.

First, export your data to spaCy's format with Prodigy - while we aren't training a spaCy model, the data will be easy to convert for HuggingFace.

```
python -m prodigy data-to-spacy cooking-openai data/ -ns 0
```

This will create the file `data/train.spacy` with your annotated documents. You'll see a warning about not creating evaluation data, but that's OK because our training script will create it.

To train the model, run the training script like this:

```
python scripts/train_hf_ner.py data/train.spacy ner-model
```

This will run for a while and train your first model. With just 100 annotations performance may not be great, but you should see it improve over each epoch, which is a sign that your data is consistent and you're on the right track. The resulting model will be saved to the `ner-model/` directory.

From here all you have to do is continue to iterate on your model until you're happy with it.
