from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
)

from ...models import TrainingDataset, TrainingDirs


@dataclass
class GPT2TrainerParams:
    dirs: TrainingDirs
    dataset: TrainingDataset
    tokenizer: Path | str
    num_train_epochs: int
    per_device_batch_size: int = 32
    context_length: int = 128


class GPT2Trainer:
    params: GPT2TrainerParams

    def __init__(self, params: GPT2TrainerParams) -> None:
        self.params = params

    def train(self) -> None:
        tokenizer = GPT2TokenizerFast.from_pretrained(str(self.params.tokenizer))

        config = GPT2Config.from_pretrained(
            "gpt2",
            vocab_size=len(tokenizer),
            n_ctx=self.params.context_length,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        model = GPT2LMHeadModel(config)
        model = model.to("cuda") if torch.cuda.is_available() else model

        tokenizer.pad_token = tokenizer.eos_token
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

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

        args = TrainingArguments(
            evaluation_strategy="epoch",
            output_dir=str(self.params.dirs.checkpoint),
            num_train_epochs=self.params.num_train_epochs,
            per_device_train_batch_size=self.params.per_device_batch_size,
            per_device_eval_batch_size=self.params.per_device_batch_size,
            logging_dir=str(self.params.dirs.logging),
            fp16=True if torch.cuda.is_available() else False,
            report_to=["tensorboard"],
        )

        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=args,
            data_collator=data_collator,
            train_dataset=dataset_dict["train"],
            eval_dataset=dataset_dict["test"],
        )

        trainer.train()
        trainer.save_model(str(self.params.dirs.model))

        metrics = trainer.evaluate(eval_dataset=dataset_dict["test"])
        trainer.save_metrics("all", metrics)
