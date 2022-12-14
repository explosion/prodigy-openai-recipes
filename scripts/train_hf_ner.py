import spacy
from spacy.tokens import DocBin
from pathlib import Path

from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import evaluate
from typing import List, Union
import numpy as np

import typer


def spacy2hf(fname: Union[str, Path], tokenizer: AutoTokenizer) -> List[BatchEncoding]:
    """Given a path to a .spacy file and an HF tokenizer, return HF tokens with
    NER labels.
    """
    infile = fname
    nlp = spacy.blank("en")
    db = DocBin().from_disk(infile)

    hfdocs = []
    # first, make ids for all labels
    label2id = {}
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
                label2id[ent_label] = len(label2id) + 1
            labels.append(ent_label)

        # now do the hf tokenization
        # maybe need truncate=true?
        tokens_hf = tokenizer(toks, truncation=True, is_split_into_words=True)
        labels_hf = []

        for word_id in tokens_hf.word_ids():
            if word_id is None:
                # for things like [CLS]
                labels_hf.append(-100)
            elif labels[word_id] == "O":
                labels_hf.append(0)
            else:
                # XXX it seems it's common to only do this for the first sub-token.
                # May want to add that as an option.
                label = label2id[labels[word_id]]
                labels_hf.append(label)
        tokens_hf["labels"] = labels_hf

        hfdocs.append(tokens_hf)

    return hfdocs, label2id


def train_bert(base_model, tokenizer, label_list, train_data, test_data):
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
        num_train_epochs=30,
        weight_decay=1e-5,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)
    seqeval = evaluate.load("seqeval")

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
    # TODO parametrize
    trainer.save_model("un-ner.model")


def run_everything(infile: str, base: str):
    tokenizer = AutoTokenizer.from_pretrained(base)
    dataset, label2id = spacy2hf(infile, tokenizer)
    id2label = {v: k for k, v in label2id.items()}
    id2label[0] = "O"
    train_bert(base, tokenizer, id2label, dataset, dataset)


if __name__ == "__main__":
    app = typer.Typer(name="Convert spaCy to HF NER data", no_args_is_help=True)
    app.command("run_everything")(run_everything)
    app()
