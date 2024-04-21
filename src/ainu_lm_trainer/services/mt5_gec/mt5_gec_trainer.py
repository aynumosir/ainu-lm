from dataclasses import dataclass

import torch
from transformers import (
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    MT5ForConditionalGeneration,
    MT5TokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from ...models import TrainingDataset, TrainingDirs


@dataclass
class MT5GECTrainerParams:
    dirs: TrainingDirs
    dataset: TrainingDataset
    num_train_epochs: int
    per_device_batch_size: int = 32
    context_length: int = 128


TASK_PREFIX = "aynu itak pirkare: "


class MT5GECTrainer:
    params: MT5GECTrainerParams

    def __init__(self, params: MT5GECTrainerParams) -> None:
        self.params = params

    # https://huggingface.co/docs/transformers/en/tasks/summarization#preprocess
    def preprocess_function(self, tokenizer: MT5TokenizerFast, examples: dict) -> dict:
        inputs = tokenizer(
            [TASK_PREFIX + text for text in examples["text"]],
            max_length=self.params.context_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = tokenizer(
            text_target=examples["labels"],
            max_length=self.params.context_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        inputs["labels"] = labels["input_ids"]
        return inputs

    def train(self) -> None:
        tokenizer = MT5TokenizerFast.from_pretrained("google/mt5-base")

        model = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")
        model = model.to("cuda") if torch.cuda.is_available() else model

        dataset = self.params.dataset.get_dataset_raw()
        dataset = dataset.map(
            lambda examples: self.preprocess_function(tokenizer, examples),
            batched=True,
        )
        dataset_dict = dataset.train_test_split(test_size=0.1)
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        training_args = Seq2SeqTrainingArguments(
            save_strategy="epoch",
            evaluation_strategy="epoch",
            output_dir=str(self.params.dirs.checkpoint),
            num_train_epochs=self.params.num_train_epochs,
            per_device_train_batch_size=self.params.per_device_batch_size,
            per_device_eval_batch_size=self.params.per_device_batch_size,
            # とくに根拠はないけどT5の事前学習でうまく行った数値
            learning_rate=3e-4,
            weight_decay=0.01,
            predict_with_generate=True,
            # https://dev.classmethod.jp/articles/huggingface-usage-early-stopping/#toc-5
            load_best_model_at_end=True,
            logging_dir=str(self.params.dirs.logging),
            report_to=["tensorboard"],
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset_dict["train"],
            eval_dataset=dataset_dict["test"],
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        trainer.train()
        trainer.save_model(str(self.params.dirs.model))

        metrics = trainer.evaluate(eval_dataset=dataset_dict["test"])
        trainer.save_metrics("all", metrics)
