from pathlib import Path
from typing import List, Optional, Union

import evaluate
import numpy as np
import spacy.util
import typer
from spacy.tokens import DocBin
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification, Trainer,
                          TrainingArguments)
from transformers.tokenization_utils_base import BatchEncoding

app = typer.Typer()

# This can't be imported like a normal library
seqeval = evaluate.load("seqeval")


def spacy2hf(
    nlp_config: Path, fname: Union[str, Path], label2id: dict, tokenizer: AutoTokenizer
) -> List[BatchEncoding]:
    """Given a path to a .spacy file, a label mapping, and an HF tokenizer,
    return HF tokens with NER labels.
    """

    infile = fname
    config = spacy.util.load_config(nlp_config)
    nlp = spacy.util.load_model_from_config(config)
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


@app.command(
    "train_hf_ner", context_settings={"allow_extra_args": False}
)
def train_hf_ner(
    # fmt: off
    config_file: Path = typer.Argument(..., help="Path to nlp config file", exists=True, allow_dash=False),
    train_file: Path = typer.Argument(..., help="Binary .spacy file containing training data", exists=True, allow_dash=False),
    dev_file: Path = typer.Argument(..., help="Binary .spacy file containing dev evaluation data", exists=True, allow_dash=False),
    output_path: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory to store trained pipeline in"),
    base: str = typer.Option("distilbert-base-uncased", "--base", "-b", help="Base transformer model to start from")
):
    """Fine-tune a HuggingFace NER model using a .spacy file as input."""
    # prep the data
    tokenizer = AutoTokenizer.from_pretrained(base)
    label2id = {"O": 0}
    train = spacy2hf(config_file, train_file, label2id, tokenizer)
    test = spacy2hf(config_file, dev_file, label2id, tokenizer)
    # handle the mapping
    id2label = {v: k for k, v in label2id.items()}
    # actually train
    trainer = train_ner(base, tokenizer, id2label, train, test)
    if output_path:
        trainer.save_model(output_path)


if __name__ == "__main__":
    app()
