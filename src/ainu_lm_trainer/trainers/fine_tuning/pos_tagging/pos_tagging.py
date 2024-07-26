import evaluate
import numpy as np
import torch
from transformers import (
    EvalPrediction,
    RobertaForTokenClassification,
    RobertaTokenizerFast,
    Trainer,
    TrainingArguments,
)

from ....config import DatasetConfig, FineTuningConfig, TrainingConfig, WorkspaceConfig
from .label_names import label_names

seqeval = evaluate.load("seqeval")

CONTEXT_LENGTH = 128


def tokenize_and_align_labels(tokenizer: RobertaTokenizerFast, examples: dict) -> dict:
    tokenized_inputs = tokenizer(
        examples["words"],
        truncation=True,
        max_length=CONTEXT_LENGTH,
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


# https://huggingface.co/docs/transformers/en/tasks/token_classification#evaluate
def compute_metrics(p: EvalPrediction) -> dict:
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]  # noqa: E741
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_names[l] for (p, l) in zip(prediction, label) if l != -100]  # noqa: E741
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def train(
    config_dataset: DatasetConfig,
    config_training: TrainingConfig,
    config_workspace: WorkspaceConfig,
    config_fine_tuning: FineTuningConfig,
) -> None:
    tokenizer = RobertaTokenizerFast.from_pretrained(
        config_fine_tuning.base_tokenizer,
        add_prefix_space=True,
    )

    model = RobertaForTokenClassification.from_pretrained(
        config_fine_tuning.base_model,
        id2label={i: label for i, label in enumerate(label_names)},
        label2id={label: i for i, label in enumerate(label_names)},
    )
    model = model.to("cuda") if torch.cuda.is_available() else model

    dataset_dict = config_dataset.load()
    dataset_dict = dataset_dict.map(
        lambda examples: tokenize_and_align_labels(tokenizer, examples),
        batched=True,
    )

    training_args = TrainingArguments(
        output_dir=str(config_workspace.checkpoint_dir),
        logging_dir=str(config_workspace.logging_dir),
        report_to=["tensorboard"],
    )
    training_args = config_training.extend(training_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["test"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(config_workspace.model_dir))

    if config_training.push_to_hub:
        model.push_to_hub(training_args.hub_model_id)
        tokenizer.push_to_hub(training_args.hub_model_id)
