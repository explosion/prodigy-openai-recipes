# openai-prodigy-recipes

An internal repo to share code for OpenAI recipes.

## Running the NER demo. 

First make sure the non-Prodigy dependencies are installed. 

```
python -m pip install ner-requirements.txt
```

Then you can run the experiment via: 

```
python -m prodigy ner.openai.correct openai-ner-demo examples.jsonl en -l person,place,company -F openai-ner.py
```

If you'd like to understand the prompts and reponses, feel free to use the `--verbose` flag. It'll give extra output that looks like below: 

```
╭─────────────────────────── Prompt to OpenAI ───────────────────────────╮
│ From the text below, extract the following entities in the following   │
│ format:                                                                │
│ Person: <comma-separated list of each person mentioned>                │
│ Place: <comma-separated list of each place mentioned>                  │
│ Company: <comma-separated list of each company mentioned>              │
│                                                                        │
│ Text:                                                                  │
│ """                                                                    │
│ Vincent D. Warmerdam lives in Haarlem with two cats. They are called   │
│ Sok and Noa.                                                           │
│ """                                                                    │
│                                                                        │
│ Answer:                                                                │
│                                                                        │
╰────────────────────────────────────────────────────────────────────────╯
╭───────────────────────── Response from OpenAI ─────────────────────────╮
│ Person: Vincent D. Warmerdam                                           │
│ Place: Haarlem                                                         │
│ Company:                                                               │
╰────────────────────────────────────────────────────────────────────────╯
```