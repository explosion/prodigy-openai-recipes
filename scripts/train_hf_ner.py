from pathlib import Path
import random
from typing import List, Union, Tuple

import spacy
from spacy.tokens import DocBin, Doc
from thinc.api import fix_random_seed

from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import evaluate
import numpy as np

import typer

# This can't be imported like a normal library
seqeval = evaluate.load("seqeval")


def split_data(docs: List[BatchEncoding], split: float = 0.8):
    """Given list of HF docs, split for train/test."""

    # This could be done to the spaCy docs instead, but it's easier to do it to
    # the HF docs because of the need to create a label/id mapping.
    random.shuffle(docs)

    train = []
    test = []

    thresh = int(len(docs) * split)

    for ii, doc in enumerate(docs):
        if ii < thresh:
            train.append(doc)
        else:
            test.append(doc)
    return train, test


def spacy2hf(
    fname: Union[str, Path], label2id: dict, tokenizer: AutoTokenizer
) -> List[BatchEncoding]:
    """Given a path to a .spacy file, a label mapping, and an HF tokenizer,
    return HF tokens with NER labels.
    """

    infile = fname
    nlp = spacy.blank("en")
    db = DocBin().from_disk(infile)

    hfdocs = []
    # first, make ids for all labels
    for doc in db.get_docs(nlp.vocab):
        labels = []
        toks = []
        for tok in doc:
            toks.append(tok.text)
            if tok.ent_type == 0:
                labels.append("O")
                continue
            ent_label = f"{tok.ent_iob_}-{tok.ent_type_}"
            if ent_label not in label2id:
                label2id[ent_label] = len(label2id)
            labels.append(ent_label)

        # now do the hf tokenization
        tokens_hf = tokenizer(toks, truncation=True, is_split_into_words=True)
        labels_hf = []

        for word_id in tokens_hf.word_ids():
            if word_id is None:
                # for things like [CLS]
                labels_hf.append(-100)
            else:
                # The docs note it's common to assign -100 to subwords after the
                # first inside an entity, but this does the simpler thing.
                label = label2id[labels[word_id]]
                labels_hf.append(label)
        tokens_hf["labels"] = labels_hf

        hfdocs.append(tokens_hf)

    return hfdocs


def build_compute_metrics(label_list):
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    return compute_metrics


def train_ner(base_model, tokenizer, label_list, train_data, test_data):
    """Fine-tune an existing HF model."""
    model = AutoModelForTokenClassification.from_pretrained(
        base_model, num_labels=len(label_list)
    )

    batch_size = 16

    args = TrainingArguments(
        f"test-ner",
        evaluation_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=20,
        weight_decay=1e-5,
        disable_tqdm=True,
        # specify the optimizer to avoid a deprecation warning
        optim="adamw_torch",
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)
    compute_metrics = build_compute_metrics(label_list)

    trainer = Trainer(
        model,
        args,
        train_dataset=train_data,
        eval_dataset=test_data,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate()
    return trainer


def train_hf_ner(
    train_file: str, dev_file: str, outdir: str, base: str = "distilbert-base-uncased"
):
    """Fine-tune a HuggingFace NER model using a .spacy file as input."""
    # prep the data
    tokenizer = AutoTokenizer.from_pretrained(base)
    label2id = {"O": 0}
    train = spacy2hf(train_file, label2id, tokenizer)
    test = spacy2hf(dev_file, label2id, tokenizer)
    # handle the mapping
    id2label = {v: k for k, v in label2id.items()}
    # actually train
    trainer = train_ner(base, tokenizer, id2label, train, test)
    trainer.save_model(outdir)


if __name__ == "__main__":
    app = typer.Typer(
        name="Train a HuggingFace NER model from spaCy formatted data",
        no_args_is_help=True,
    )
    app.command("train_hf_ner")(train_hf_ner)
    app()
