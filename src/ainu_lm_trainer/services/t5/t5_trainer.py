from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    T5TokenizerFast,
    Trainer,
    TrainingArguments,
)

from ...models import TrainingDataset, TrainingDirs
from .t5_data_collator import DataCollatorForT5


@dataclass
class T5TrainerParams:
    dirs: TrainingDirs
    dataset: TrainingDataset
    tokenizer: Path | str
    num_train_epochs: int
    per_device_batch_size: int = 32
    context_length: int = 128


class T5Trainer:
    params: T5TrainerParams

    def __init__(self, params: T5TrainerParams) -> None:
        self.params = params

    def train(self) -> None:
        tokenizer = T5TokenizerFast.from_pretrained(str(self.params.tokenizer))

        config = T5Config.from_pretrained("google-t5/t5-base")

        model = T5ForConditionalGeneration(config)
        model = model.to("cuda") if torch.cuda.is_available() else model

        data_collator = DataCollatorForT5(tokenizer=tokenizer)

        dataset = self.params.dataset.get_dataset()
        dataset = dataset.map(
            lambda examples: tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.params.context_length,
                padding="max_length",
                return_tensors="pt",
            ),
            batched=True,
        )
        dataset_dict = dataset.train_test_split(test_size=0.1)

        training_args = TrainingArguments(
            evaluation_strategy="epoch",
            output_dir=str(self.params.dirs.checkpoint),
            num_train_epochs=self.params.num_train_epochs,
            per_device_train_batch_size=self.params.per_device_batch_size,
            per_device_eval_batch_size=self.params.per_device_batch_size,
            logging_dir=str(self.params.dirs.logging),
            report_to=["tensorboard"],
        )

        trainer = Trainer(
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
