<a href="https://explosion.ai"><img src="https://explosion.ai/assets/img/logo.svg" width="125" height="125" align="right" /></a>

## Archival notice 

The recipes in this repository have since moved to [Prodigy](https://prodi.gy/) and are being maintained there. You
can learn more by checking out the [large language models section](https://prodi.gy/docs/large-language-models) 
on the docs. 

# Prodigy OpenAI recipes

This repository contains example code on how to combine **zero- and few-shot learning
with a small annotation effort** to obtain a **high-quality dataset with maximum efficiency**. Specifically, we use large language models available from [OpenAI](https://openai.com) to provide us with an initial set of predictions,
then spin up a [Prodigy](https://prodi.gy) instance on our local machine
to go through these predictions and curate them. This allows us to obtain a
gold-standard dataset pretty quickly, and train a smaller, supervised model that fits
our exact needs and use-case.

![](https://user-images.githubusercontent.com/13643239/208497043-178beb47-f7c6-4b3e-a253-9e12e2f0c807.png)

https://user-images.githubusercontent.com/13643239/208504034-0ab6bcbe-6d2b-415d-8257-233f2074ba31.mp4

## ⏳ Setup and Install

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

## 📋 Named-entity recognition (NER)

### `ner.openai.correct`: NER annotation with zero- or few-shot learning

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
| `filepath`              | Path | Path to `.jsonl` data to annotate. The data should at least contain a `"text"` field.                                                           |                                 |
| `labels`                | str  | Comma-separated list defining the NER labels the model should predict.                                                                          |                                 |
| `--lang`, `-l`          | str  | Language of the input data - will be used to obtain a relevant tokenizer.                                                                       | `"en"`                          |
| `--segment`, `-S`       | bool | Flag to set when examples should be split into sentences. By default, the full input article is shown.                                          | `False`                         |
| `--model`, `-m`         | str  | GPT-3 model to use for initial predictions.                                                                                                     | `"text-davinci-003"`            |
| `--prompt_path`, `-p`   | Path | Path to the `.jinja2` [prompt template](templates).                                                                                             | `./templates/ner_prompt.jinja2` |
| `--examples-path`, `-e` | Path | Path to examples to help define the task. The file can be a .yml, .yaml or .json. If set to `None`, zero-shot learning is applied.              | `None`                          |
| `--max-examples`, `-n`  | int  | Max number of examples to include in the prompt to OpenAI. If set to 0, zero-shot learning is always applied, even when examples are available. | 2                               |
| `--batch-size`, `-b`    | int  | Batch size of queries to send to the OpenAI API.                                                                                                | 10                              |
| `--verbose`, `-v`       | bool | Flag to print extra information to the terminal.                                                                                                | `False`                         |

#### Example usage

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
python -m prodigy ner.openai.correct my_ner_data ./data/reddit_r_cooking_sample.jsonl "dish,ingredient,equipment" -p ./templates/ner_prompt.jinja2 -e ./examples/ner.yaml -n 2 -F ./recipes/openai_ner.py
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
that are sent to the OpenAI API as part of the prompt.

> **Note**  
> Because Prodigy batches these requests, the prompt will be updated with a slight
> delay, after the next batch of prompts is sent to OpenAI. You can experiment
> with making the batch size (`--batch-size` or `-b`) smaller to have the change
> come into effect sooner, but this might negatively impact the speed of the
> annotation workflow.

### `ner.openai.fetch`: Fetch examples up-front

The `ner.openai.correct` recipe fetches examples from OpenAI while annotating, but we've also included a recipe that can fetch a large batch of examples upfront.

```bash
python -m prodigy ner.openai.fetch input_data.jsonl predictions.jsonl "dish,ingredient,equipment" -F ./recipes/ner.py
```

This will create a `predictions.jsonl` file that can be loaded with the [`ner.manual`](https://prodi.gy/docs/recipes#ner-manual) recipe.

Note that the OpenAI API might return "429 Too Many Request" errors when requesting too much data at once - in this case it's best to ensure you only request
100 or so examples at a time.

### Exporting the annotations and training an NER model

After you've curated a set of predictions, you can export the results with [`db-out`](https://prodi.gy/docs/recipes#db-out):

```bash
python -m prodigy db-out my_ner_data  > ner_data.jsonl
```

The format of the exported annotations contains all the data you need to train a smaller model downstream. Each example
in the dataset contains the original text, the tokens, span annotations denoting the entities, etc.

You can also export the data to spaCy's [binary format](https://spacy.io/api/data-formats#training), using [`data-to-spacy`](https://prodi.gy/docs/recipes#data-to-spacy). This format lets you load in the annotations as spaCy `Doc` objects, which can be convenient for further conversion. The `data-to-spacy` command also makes it easy to train an NER model with spaCy. First you export the data, specifying the train data as 20% of the total:

```bash
python -m prodigy data-to-spacy ./data/annotations/ --ner my_ner_data -es 0.2
```

Then you can train a model with spaCy or [Prodigy](https://prodi.gy/docs/recipes/#training):

```bash
python -m spacy train ./data/annotations/config.cfg --paths.train ./data/annotations/train.spacy --paths.dev ./data/annotations/dev.spacy -o ner-model
```

This will save a model to the `ner-model/` directory.

We've also included an experimental script to load in the `.spacy` binary format and train a model with the HuggingFace `transformers` library. You can use the same data you just exported and run the script like this:

```bash
# First you need to install the HuggingFace library and requirements
pip install -r requirements_train.txt
python ./scripts/train_hf_ner.py ./data/annotations/train.spacy ./data/annotations/dev.spacy -o hf-ner-model
```

The resulting model will be saved to the `hf-ner-model/` directory.

## 📋 Text categorization (Textcat)

### `textcat.openai.correct`: Textcat annotation with zero- or few-shot learning

This recipe enables us to classify texts faster with the help of a large
language model. It also provides a "reason" to explain why a particular label
was chosen. 

```bash
python -m prodigy textcat.openai.correct dataset filepath labels [--options] -F ./recipes/openai_textcat.py
```

| Argument                    | Type | Description                                                                                                                                     | Default                             |
| --------------------------- | ---- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------- |
| `dataset`                   | str  | Prodigy dataset to save annotations to.                                                                                                         |                                     |
| `filepath`                  | Path | Path to `.jsonl` data to annotate. The data should at least contain a `"text"` field.                                                           |                                     |
| `labels`                    | str  | Comma-separated list defining the text categorization labels the model should predict.                                                          |                                     |
| `--lang`, `-l`              | str  | Language of the input data - will be used to obtain a relevant tokenizer.                                                                       | `"en"`                              |
| `--segment`, `-S`           | bool | Flag to set when examples should be split into sentences. By default, the full input article is shown.                                          | `False`                             |
| `--model`, `-m`             | str  | GPT-3 model to use for initial predictions.                                                                                                     | `"text-davinci-003"`                |
| `--prompt-path`, `-p`       | Path | Path to the `.jinja2` [prompt template](templates).                                                                                             | `./templates/textcat_prompt.jinja2` |
| `--examples-path`, `-e`     | Path | Path to examples to help define the task. The file can be a .yml, .yaml or .json. If set to `None`, zero-shot learning is applied.              | `None`                              |
| `--max-examples`, `-n`      | int  | Max number of examples to include in the prompt to OpenAI. If set to 0, zero-shot learning is always applied, even when examples are available. | 2                                   |
| `--batch-size`, `-b`        | int  | Batch size of queries to send to the OpenAI API.                                                                                                | 10                                  |
| `--exclusive-classes`, `-E` | bool | Flag to make the classification task exclusive.                                                                                                 | `False`                             |
| `--verbose`, `-v`           | bool | Flag to print extra information to the terminal.                                                                                                | `False`                             |


#### Example usage

The `textcat` recipes can be used for binary, multiclass, and multilabel text
categorization. You can set this by passing the appropriate number of labels in
the `--labels` parameter; for example, passing a single label turns it into
binary classification and so on. We will talk about each one in the proceeding
sections.

##### Binary text categorization

Suppose we want to know if a particular Reddit comment talks about a food
recipe.  We'll send the text to GPT-3 and provide a prompt that instructs the
predictions we want. 

```
From the text below, determine wheter or not it contains a recipe. If it is a 
recipe, answer "accept." If it is not a recipe, answer "reject."

Your answer should only be in the following format:
answer: <string>
reason: <string>

Text:
```

For binary classification, we want GPT-3 to return "accept" if a given text is a
food recipe and "reject" otherwise. GPT-3's suggestion is then displayed
prominently in the UI. We can press the <kbd>ACCEPT</kbd> (check mark) button to
include the text as a positive example or press the <kbd>REJECT</kbd> (cross
mark) button if it is a negative example.


```sh
python -m prodigy textcat.openai.correct my_binary_textcat_data data/reddit_r_cooking_sample.jsonl --labels recipe -F recipes/openai_textcat.py
```

<img src="https://user-images.githubusercontent.com/12949683/214230166-ee492fe5-04da-4b93-9590-b5ef23ce488d.png" width="600"/>


##### Multilabel and multiclass text categorization

Now, suppose we want to classify Reddit comments as a recipe, a feedback, or a
question. We can write the following prompt: 

```
Classify the text below to any of the following labels: recipe, feedback, question.
The task is exclusive, so only choose one label from what I provided.

Your answer should only be in the following format:
answer: <string>
reason: <string>

Text:
```

Then, we can use this recipe to handle multilabel and multiclass cases by
passing the three labels to the `--labels` parameter. We should also set the
`--exclusive-classes` flag to render a single-choice UI:

```sh
python -m prodigy textcat.openai.correct my_multi_textcat_data data/reddit_r_cooking_sample.jsonl \
    --labels recipe,feedback,question \
    --exclusive-classes \
    -F recipes/openai_textcat.py
```

<img src="https://user-images.githubusercontent.com/12949683/214230269-bdacbbc8-8edc-4be5-8334-0b7eaf4712e0.png" width="600" />

### Writing templates

We write these prompts as a .jinja2 template that can also take in examples for
few-shot learning. You can create your own [template](templates) and provide it
to the recipe with the `--prompt-path` or `-p` option.  Additionally, with
`--examples-path` or `-e` you can set the file path of a .y(a)ml or .json file
that contains additional examples. You can also add context in these examples as
we observed it to improve the output:

```bash
python -m prodigy textcat.openai.correct my_binary_textcat_data \
    ./data/reddit_r_cooking_sample.jsonl \
    --labels recipe \
    --prompt-path ./templates/textcat_prompt.jinja2 \
    --examples-path ./examples/textcat_binary.yaml -n 2 \
    -F ./recipes/openai_textcat.py
```

Similar to the NER recipe, this recipe also converts the predictions into an 
annotation task that can be rendered with Prodigy. For binary classification, we
use the [`classification`](https://prodi.gy/docs/api-interfaces#classification)
interface with custom HTML elements, while for multilabel or multiclass text
categorization, we use the
[`choice`](https://prodi.gy/docs/api-interfaces#choice) annotation interface.
Notice that we include the original prompt and the OpenAI response in the UI. 


Lastly, you can use the `--verbose` or `-v` flag to show the exact prompt and
response on the terminal. Note that because the requests to the API are batched,
you might have to scroll back a bit to find the current prompt.


### Interactively tune the prompt examples

Similar to the NER recipes, you can also steer the predictions in the right
direction by correcting the example and then selecting the small "flag" icon in
the top right of the Prodigy UI:

<img src="https://user-images.githubusercontent.com/12949683/214780178-96be66e4-8a02-4820-a51e-3e6040bcddc1.png" width="600"/>

Once you hit the <kbd>accept</kbd> button on the Prodigy interface, the flagged
example will be picked up and added to the few-shot examples sent to the OpenAI
API as part of the prompt.

> **Note**  
> Because Prodigy batches these requests, the prompt will be updated with a slight
> delay, after the next batch of prompts is sent to OpenAI. You can experiment
> with making the batch size (`--batch-size` or `-b`) smaller to have the change
> come into effect sooner, but this might negatively impact the speed of the
> annotation workflow.

### `textcat.openai.fetch`: Fetch text categorization examples up-front

The `textcat.openai.fetch` recipe allows us to fetch a large batch of examples
upfront. This is helpful when you are with a highly-imbalanced data and interested
only in rare examples.

```bash
python -m prodigy textcat.openai.fetch input_data.jsonl predictions.jsonl --labels Recipe -F ./recipes/openai_textcat.py
```

This will create a `predictions.jsonl` file that can be loaded with the
[`textcat.manual`](https://prodi.gy/docs/recipes#textcat-manual) recipe.

Note that the OpenAI API might return "429 Too Many Request" errors when
requesting too much data at once - in this case it's best to ensure you only
request 100 or so examples at a time and take a look at the [API's rate
limits](https://help.openai.com/en/articles/5955598-is-api-usage-subject-to-any-rate-limits).

#### Working with imbalanced data

The `textcat.openai.fetch` recipe is suitable for working with datasets where
there is severe class imbalance. Usually, you'd want to find examples of the
rare class rather than annotating a random sample. From there, you want to
upsample them to train a decent model and so on.

This is where large language models like OpenAI might help.

Using the [Reddit r/cooking dataset](data), we prompted OpenAI to
look for comments that resemble a food recipe. Instead of annotating 10,000
examples, we ran `textcat.openai.fetch` and obtained 145 positive classes. Out
of those 145 examples, 114 turned out to be true positives (79% precision). We
then checked 1,000 negative examples and found 12 false negative cases (98%
recall). 

Ideally, once we fully annotated the dataset, we can train a supervised model
that is better to use than relying on zero-shot predictions for production. The
running cost is low and it's easier to manage.

### Exporting the annotations and training a text categorization model

After you've curated a set of predictions, you can export the results with
[`db-out`](https://prodi.gy/docs/recipes#db-out):

```bash
python -m prodigy db-out my_textcat_data  > textcat_data.jsonl
```

The format of the exported annotations contains all the data you need to train a
smaller model downstream. Each example in the dataset contains the original
text, the tokens, span annotations denoting the entities, etc.

You can also export the data to spaCy's [binary
format](https://spacy.io/api/data-formats#training), using
[`data-to-spacy`](https://prodi.gy/docs/recipes#data-to-spacy). This format lets
you load in the annotations as spaCy `Doc` objects, which can be convenient for
further conversion. The `data-to-spacy` command also makes it easy to train a
text categorization model with spaCy. First you export the data, specifying the
train data as 20% of the total:

```bash
# For binary textcat
python -m prodigy data-to-spacy ./data/annotations/ --textcat my_textcat_data -es 0.2
# For multilabel textcat
python -m prodigy data-to-spacy ./data/annotations/ --textcat-multilabel my_textcat_data -es 0.2
```
Then you can train a model with spaCy or [Prodigy](https://prodi.gy/docs/recipes/#training):

```bash
python -m spacy train ./data/annotations/config.cfg --paths.train ./data/annotations/train.spacy --paths.dev ./data/annotations/dev.spacy -o textcat-model
```

This will save a model to the `textcat-model/` directory.

## 📋 Terms 

### `terms.openai.fetch`: Fetch phrases and terms based on a query

This recipe generates terms and phrases obtained from a large language model. These
terms can be curated and turned into patterns files, which can help with downstream annotation tasks. 

```bash
python -m prodigy terms.openai.fetch query filepath [--options] -F ./recipes/openai_terms.py
```

|        Argument       | Type  | Description                                         | Default                         |
|:---------------------:|-------|-----------------------------------------------------|---------------------------------|
| `query`               | str   | Query to send to OpenAI                             |                                 |
| `output_path`         | Path  | Path to save the output                             |                                 |
| `--seeds`,`-s`        | str   | One or more comma-separated seed phrases.           | `""`                            |
| `--n`,`-n`            | int   | Minimum number of items to generate                 | `100`                           |
| `--model`, `-m`       | str   | GPT-3 model to use for completion                   | `"text-davinci-003"`            |
| `--prompt-path`, `-p` | Path  | Path to jinja2 prompt template                      | `templates/terms_prompt.jinja2` |
| `--verbose`,`-v`      | bool  | Print extra information to terminal                 | `False`                         |
| `--resume`, `-r`      | bool  | Resume by loading in text examples from output file | `False`                         |
| `--progress`,`-pb`    | bool  | Print progress of the recipe.                       | `False`                         |
| `--temperature`,`-t`  | float | OpenAI temperature param                            | `1.0`                           |
| `--top-p`, `--tp`     | float | OpenAI top_p param                                  | `1.0`                           |
| `--best-of`, `-bo`    | int   | OpenAI best_of param"                               | `10`                            |
| `--n-batch`,`-nb`     | int   | OpenAI batch size param                             | `10`                            |
| `--max-tokens`, `-mt` | int   | Max tokens to generate per call                     | `100`                           |

#### Example usage

Suppose you're interested in detecting skateboard tricks in text, then you might want to start
with a term list of known tricks. You might want to start with the following query:

```bash
# Base behavior, fetch at least 100 terms/phrases
python -m prodigy terms.openai.fetch "skateboard tricks" tricks.jsonl --n 100 --prompt-path templates/terms_prompt.jinja2 -F recipes/openai_terms.py
```

This will generate a prompt to OpenAI that asks to try and generate at least 100 examples of "skateboard tricks".
There's an upper limit to the amount of tokens that can be generated by OpenAI, but this recipe will try and keep
collecting terms until it reached the amount specified. 

You can choose to make the query more elaborate if you want to try to be more precise, but you can alternatively
also choose to add some seed terms via `--seeds`. These will act as starting examples to help steer OpenAI 
in the right direction.

```bash
# Base behavior but with seeds
python -m prodigy terms.openai.fetch "skateboard tricks" tricks.jsonl --n 100 --seeds "kickflip,ollie" --prompt-path templates/terms_prompt.jinja2 -F recipes/openai_terms.py
```

Collecting many examples can take a while, so it can be helpful to show the progress, via `--progress` 
as requests are sent. 

```bash
# Adding progress output as we wait for 500 examples
python -m prodigy terms.openai.fetch "skateboard tricks" tricks.jsonl --n 500 --progress --seeds "kickflip,ollie" --prompt-path templates/terms_prompt.jinja2 -F recipes/openai_terms.py
```

After collecting a few examples, you might want to generate more. You can choose to continue from a
previous output file. This will effectively re-use those examples as seeds for the prompt to OpenAI.

```bash
# Use the `--resume` flag to re-use previous examples
python -m prodigy terms.openai.fetch "skateboard tricks" tricks.jsonl --n 50 --resume --prompt-path templates/terms_prompt.jinja2 -F recipes/openai_terms.py
```

When the recipe is done, you'll have a `tricks.jsonl` file that has contents that look like this: 

```json
{"text":"pop shove it","meta":{"openai_query":"skateboard tricks"}}
{"text":"switch flip","meta":{"openai_query":"skateboard tricks"}}
{"text":"nose slides","meta":{"openai_query":"skateboard tricks"}}
{"text":"lazerflip","meta":{"openai_query":"skateboard tricks"}}
{"text":"lipslide","meta":{"openai_query":"skateboard tricks"}}
...
```

### Towards Patterns 

You now have a `tricks.jsonl` file on disk that contains skateboard tricks, but you cannot
assume that all of these will be accurate. The next step would be to review the terms and you
can use the [`textcat.manual`](https://prodi.gy/docs/recipes/#textcat-manual) recipe that comes
with Prodigy for that. 

```bash
# The tricks.jsonl was fetched from OpenAI beforehand
python -m prodigy textcat.manual skateboard-tricks-list tricks.jsonl --label skateboard-tricks
```

This generates an interface that looks like this: 

<img src="https://user-images.githubusercontent.com/1019791/212869305-58f1d087-e036-4eab-a818-df80aab68ce8.png" width="600" />

You can manually accept or reject each example and when you're done annotating you can export
the annotated text into a patterns file via the [`terms.to-patterns`](https://prodi.gy/docs/recipes/#terms-to-patterns) recipe.

```bash
# Generate a `patterns.jsonl` file.
python -m prodigy terms.to-patterns skateboard-tricks-list patterns.jsonl --label skateboard-tricks --spacy-model blank:en
```

When the recipe is done, you'll have a `patterns.jsonl` file that has contents that look like this: 

```json
{"label":"skateboard-tricks","pattern":[{"lower":"pop"},{"lower":"shove"},{"lower":"it"}]}
{"label":"skateboard-tricks","pattern":[{"lower":"switch"},{"lower":"flip"}]}
{"label":"skateboard-tricks","pattern":[{"lower":"nose"},{"lower":"slides"}]}
{"label":"skateboard-tricks","pattern":[{"lower":"lazerflip"}]}
{"label":"skateboard-tricks","pattern":[{"lower":"lipslide"}]} 
...
```

### Known Limitations 

OpenAI has a hard limit on the prompt size. You cannot have a prompt larger than 4079 tokens. Unfortunately
that means that there is a limit to the size of term lists that you can generate. The recipe will report
an error when this happens, but it's good to be aware of this limitation.

## 📋 Prompt A/B evaluation

### `ab.openai.prompts`: A/B evaluation of prompts

The goal of this recipe is to quickly allow someone to compare the quality of outputs from two prompts 
in a quantifiable and blind way. 

```bash
python -m prodigy ab.openai.prompts dataset inputs_path display_template_path prompt1_template_path prompt2_template_path [--options] -F ./recipes/openai_ab.py
```

|        Argument        | Type  | Description                                          | Default              |
|:----------------------:|-------|------------------------------------------------------|----------------------|
| `dataset`              | str   | Prodigy dataset to save answers into                 |                      |
| `inputs_path`          | Path  | Path to jsonl inputs                                 |                      |
| `display_template_path`| Path  | Template for summarizing the arguments               |                      |
| `prompt1_template_path`| Path  | Path to the first jinja2 prompt template             |                      |
| `prompt2_template_path`| Path  | Path to the second jinja2 prompt template            |                      |
| `--model`, `-m`        | str   | GPT-3 model to use for completion                    | `"text-davinci-003"` |
| `--batch-size`, `-b`   | int   | Batch size to send to OpenAI API                     | `10`                 |
| `--verbose`,`-v`       | bool  | Print extra information to terminal                  | `False`              |
| `--no-random`,`-NR`    | bool  | Don't randomize which annotation is shown as correct | `False`              |
| `--repeat`, `-r`       | int   | How often to send the same prompt to OpenAI          | `1`                  |

#### Example usage

As an example, let's try to generate humorous haikus. To do that we first need to 
construct two jinja files that represent the prompt to send to OpenAI. 

##### `templates/ab/prompt1.jinja2`

```
Write a haiku about {{topic}}.
```

##### `templates/ab/prompt2.jinja2`

```
Write an incredibly hilarious haiku about {{topic}}. So funny!
```

You can provide variables for these prompts by constructing a .jsonl file with the required
parameters. In this case we need to make sure that `{{topic}}` is accounted for. 

Here's an example `.jsonl` file that could work. 

##### `data/ab_example.jsonl`

```json
{"id": 0, "prompt_args": {"topic": "star wars"}}
{"id": 0, "prompt_args": {"topic": "kittens"}}
{"id": 0, "prompt_args": {"topic": "the python programming language"}}
{"id": 0, "prompt_args": {"topic": "maths"}}
```

> **Note**
>
> All the arguments under `prompt_args` will be passed to render the jinja templates. 
> The `id` is mandatory and can be used to identify groups in later analysis.

We're nearly ready to evaluate, but this recipe requires one final jinja2 template. 
This one won't be used to generate a prompt, but it will generate a useful title 
that reminds the annotator of the current task. Here's an example of such a template. 

##### `templates/ab/input.jinja2`

```
A haiku about {{topic}}.
```

When you put all of these templates together you can start annotating. The command below 
starts the annotation interface and also uses the `--repeat 4` option. This will ensure
that each topic will be used to generate a prompt at least 4 times.

```
python -m prodigy ab.openai.prompts haiku data/ab_example.jsonl templates/ab/input.jinja2 templates/ab/prompt1.jinja2 templates/ab/prompt2.jinja2 --repeat 5 -F recipes/openai_ab.py
```

This is what the annotation interface looks like:

![](https://user-images.githubusercontent.com/1019791/216607308-97a0b82d-03ea-4d09-ab79-0ec6b26cc033.png)

When you look at this interface you'll notice that the title template is rendered and that
you're able to pick from two options. Both options are responses from OpenAI that were
generated by the two prompt templates. You can also see the `prompt_args` rendered in 
the lower right corner of the choice menu.

From here you can annotate your favorite examples and gather data that might help you
decide on which prompt is best.

#### Results 

Once you're done annotating you'll be presented with an overview of the results. 

```
=========================== ✨  Evaluation results ===========================
✔ You preferred prompt1.jinja2

prompt1.jinja2   11
prompt2.jinja2    5
```

But you can also fetch the raw annotations from the database for further analysis. 

```
python -m prodigy db-out haiku
```

## ❓ What's next?

There’s lots of interesting follow-up experiments to this, and lots of ways to adapt the basic idea to different tasks or data sets. We’re also interested to try out different prompts. It’s unclear how much the format the annotations are requested in might change the model’s predictions, or whether there’s a shorter prompt that might perform just as well. We also want to run some end-to-end experiments.
