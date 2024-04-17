from dataclasses import dataclass

import torch
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    T5TokenizerFast,
)

from ...models import TrainingDataset, TrainingDirs


@dataclass
class T5GCETrainerParams:
    dirs: TrainingDirs
    dataset: TrainingDataset
    num_train_epochs: int
    per_device_batch_size: int = 32
    context_length: int = 128


TASK_PREFIX = "pirkare wa en=kore: "


class T5GCETrainer:
    params: T5GCETrainerParams

    def __init__(self, params: T5GCETrainerParams) -> None:
        self.params = params

    def train(self) -> None:
        tokenizer = T5TokenizerFast.from_pretrained("aynumosir/t5-base-ainu")

        model = T5ForConditionalGeneration.from_pretrained("aynumosir/t5-base-ainu")
        model = model.to("cuda") if torch.cuda.is_available() else model

        # Add `unk_id`
        tokenizer.add_special_tokens(
            {
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "additional_special_tokens": [
                    TASK_PREFIX,
                ],
            }
        )

        dataset = self.params.dataset.get_dataset_raw()

        dataset = dataset.map(
            lambda example: {
                "text": TASK_PREFIX + example["text"],
                "labels": example["labels"],
            }
        )

        dataset = dataset.map(
            lambda examples: tokenizer(
                examples["text"],
                text_target=examples["labels"],
                truncation=True,
                max_length=self.params.context_length,
                padding="max_length",
                return_tensors="pt",
            ),
            batched=True,
        )
        dataset_dict = dataset.train_test_split(test_size=0.1)
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        training_args = Seq2SeqTrainingArguments(
            evaluation_strategy="epoch",
            output_dir=str(self.params.dirs.checkpoint),
            num_train_epochs=self.params.num_train_epochs,
            per_device_train_batch_size=self.params.per_device_batch_size,
            per_device_eval_batch_size=self.params.per_device_batch_size,
            # https://huggingface.co/docs/transformers/v4.39.3/en/model_doc/t5#usage-tips:~:text=T5%20models%20need%20a%20slightly%20higher%20learning%20rate
            learning_rate=3e-4,
            logging_dir=str(self.params.dirs.logging),
            report_to=["tensorboard"],
        )

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
