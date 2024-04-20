from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5Config,
    T5ForConditionalGeneration,
    T5TokenizerFast,
)

from ...models import TrainingDataset, TrainingDirs


@dataclass
class T5GECTrainerParams:
    dirs: TrainingDirs
    tokenizer: Path | str
    dataset: TrainingDataset
    num_train_epochs: int
    per_device_batch_size: int = 32
    context_length: int = 128


TASK_PREFIX = "pirkare: "


class T5GECTrainer:
    params: T5GECTrainerParams

    def __init__(self, params: T5GECTrainerParams) -> None:
        self.params = params

    # https://huggingface.co/docs/transformers/en/tasks/summarization#preprocess
    def preprocess_function(self, tokenizer: T5TokenizerFast, examples: dict) -> dict:
        inputs = tokenizer(
            [TASK_PREFIX + text for text in examples["text"]],
            max_length=self.params.context_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = tokenizer(
            # TODO: temporarily using the same text as input
            text_target=examples["text"],
            max_length=self.params.context_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        inputs["labels"] = labels["input_ids"]
        return inputs

    def train(self) -> None:
        tokenizer = T5TokenizerFast.from_pretrained(str(self.params.tokenizer))

        config = T5Config.from_pretrained("t5-base")
        model = T5ForConditionalGeneration(config)
        model = model.to("cuda") if torch.cuda.is_available() else model

        dataset = self.params.dataset.get_dataset()
        dataset = dataset.map(
            lambda examples: self.preprocess_function(tokenizer, examples),
            batched=True,
        )
        dataset_dict = dataset.train_test_split(test_size=0.1)
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        # https://huggingface.co/learn/nlp-course/chapter7/4?fw=pt#fine-tuning-the-model
        training_args = Seq2SeqTrainingArguments(
            evaluation_strategy="epoch",
            output_dir=str(self.params.dirs.checkpoint),
            num_train_epochs=self.params.num_train_epochs,
            per_device_train_batch_size=self.params.per_device_batch_size,
            per_device_eval_batch_size=self.params.per_device_batch_size,
            learning_rate=3e-4,
            weight_decay=0.01,
            predict_with_generate=True,
            logging_dir=str(self.params.dirs.logging),
            report_to=["tensorboard"],
        )

        if torch.cuda.is_available():
            training_args.fp16 = True

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset_dict["train"],
            eval_dataset=dataset_dict["test"],
        )

        trainer.train()
        trainer.save_model(str(self.params.dirs.model))

        metrics = trainer.evaluate(eval_dataset=dataset_dict["test"])
        trainer.save_metrics("all", metrics)
