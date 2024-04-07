import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    RobertaForTokenClassification,
    RobertaTokenizerFast,
    Trainer,
    TrainingArguments,
)

seqeval = evaluate.load("seqeval")

label_names = [
    "PPX",
    "PSX",
    "PF",
    "N",
    "NL",
    "PRN",
    "NMLZ",
    "COMP",
    "PRP.N",
    "VI",
    "VT",
    "VC",
    "VD",
    "AUX",
    "ADV",
    "ADV.PP",
    "DEM",
    "PP",
    "ADV.PRT",
    "ADV.CONJ",
    "CONJ",
    "FIN.PRT",
    "NUM",
    "N.INTERR",
    "INTJ",
    "PUNCT",
]

tokenizer = RobertaTokenizerFast.from_pretrained(
    "aynumosir/roberta-base-ainu", add_prefix_space=True
)
model = RobertaForTokenClassification.from_pretrained(
    "aynumosir/roberta-base-ainu",
    id2label={i: label for i, label in enumerate(label_names)},
    label2id={label: i for i, label in enumerate(label_names)},
)

dataset = load_dataset("./treebank", split="train")
dataset_dict = dataset.train_test_split(test_size=0.1)


def tokenize_and_align_labels(examples: dict) -> dict:
    tokenized_inputs = tokenizer(
        examples["words"],
        truncation=True,
        max_length=128,
        padding="max_length",
        is_split_into_words=True,
    )
    labels = []

    for i, label in enumerate(examples["pos"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            else:
                try:
                    # ちゃんとやりたい。
                    label_ids.append(label_names.index(label[word_idx]))
                except ValueError:
                    label_ids.append(-100)
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels

    return tokenized_inputs


train_dataset = dataset_dict["train"].map(tokenize_and_align_labels, batched=True)
eval_dataset = dataset_dict["test"].map(tokenize_and_align_labels, batched=True)


# https://huggingface.co/docs/transformers/en/tasks/token_classification#evaluate
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_names[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


if __name__ == "__main__":
    args = TrainingArguments(
        output_dir="checkpoints",
        evaluation_strategy="epoch",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        logging_dir="logs",
        logging_steps=10,
        report_to="tensorboard",
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    metrics = trainer.evaluate(eval_dataset=eval_dataset)
    trainer.save_metrics("all", metrics)

    trainer.save_model("roberta-base-ainu-pos")
