# openai-prodigy-recipes

An internal repo to share code for OpenAI recipes.

## Setup and Install 

Before running any demos, you need to make sure that the `.env` file contains the right keys. 

```
OPENAI_ORG = "org-..."
OPENAI_KEY = "sk-..."
```

You'll also want to make sure the non-Prodigy dependencies are installed. 

```
python -m pip install -r requirements.txt
```

## Running the NER demos. 

We're hosting recipes that use zero/few-shot learning via OpenAI. 

![](image.png)

The recipe can take examples from a local `.jsonl` file and turn them into NER annotation prompts. These can be sent to GPT-3, hosted by OpenAI, which respond with an answer. The recipes handle the translation between Prodigy and OpenAI such that you can confirm the annotations easily from Prodigy. It's very much like using the standard [`ner.correct`](https://prodi.gy/docs/recipes#ner-correct) recipe in Prodi.gy, but we're using GPT-3 as a backend model to make predictions. 

## First Steps

For the first demo we'll use an `examples.jsonl` file that contains the following examples. 

```
{"text": "I'm a total hillfiger fanboy. Gotta love 'em jeans."}
{"text": "Levis all the way. Their jeans are solid, but their jackets are better than old navy."}
{"text": "club monaco's Super Slim Twill Pant is pretty good. i like the taper but the thighs and seat are a bit too skinny for my tastes."}
```

The goal of this dataset is to extra the fashion brands with the clothing items. So let's see what OpenAI can annotate for us! We can use the `ner.openai.correct` to send examples to their API one at a time. 

```
python -m prodigy ner.openai.correct fashion-openai examples.jsonl "brand,clothing" -F recipes/ner.py
```

Here's what the annotation interface will look like. 

![](imgs/ner-correct.png)

You'll notice that the annotation interface comes with values pre-filled, which can speed up annotation.

### Curious about the prompt?

If you're curious to see what we send to OpenAI and what we get back, you can run the recipe with the `-v` verbose flag. This will print the prompt and the response in the terminal as traffic is received. 

```
╭────────────────────────────────── Prompt to OpenAI ─────────────────────────────╮
│ From the text below, extract the following entities in the following format:    │
│ brand: <comma delimited list of strings>                                        │
│ clothing: <comma delimited list of strings>                                     │
│                                                                                 │
│ Text:                                                                           │
│ """                                                                             │
│ I'm a total hillfiger fanboy. Gotta love 'em jeans.                             │
│ """                                                                             │
│                                                                                 │
╰─────────────────────────────────────────────────────────────────────────────────╯
╭──────────────────────────────── Response from OpenAI ───────────────────────────╮
│                                                                                 │
│ brand: Hillfiger                                                                │
│ clothing: Jeans                                                                 │
╰─────────────────────────────────────────────────────────────────────────────────╯
```

This repo also provides templates that you can customise in the `/templates` folder. We use `jinja2` to populate these templates with prompts, but you can choose to create your own template and use it via the `--prompt-path` option. 

## Better Suggestions 

At some point, you might notice OpenAI make a mistake. 