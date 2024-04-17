from dataclasses import dataclass

import evaluate
import numpy as np
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

metric = evaluate.load("sacrebleu")


@dataclass
class T5GCETrainerParams:
    dirs: TrainingDirs
    dataset: TrainingDataset
    num_train_epochs: int
    per_device_batch_size: int = 32
    context_length: int = 128


TASK_PREFIX = "pirkare: "


class T5GCETrainer:
    params: T5GCETrainerParams

    def __init__(self, params: T5GCETrainerParams) -> None:
        self.params = params

    def train(self) -> None:
        tokenizer = T5TokenizerFast.from_pretrained("./models/sentencepiece")
        config = T5Config.from_pretrained("t5-small")

        model = T5ForConditionalGeneration(config)
        model = model.to("cuda") if torch.cuda.is_available() else model
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

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]

            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_preds = [pred.strip() for pred in decoded_preds]
            decoded_labels = [[label.strip()] for label in decoded_labels]

            result = metric.compute(
                predictions=decoded_preds, references=decoded_labels
            )
            return {"bleu": result["score"]}

        # https://huggingface.co/learn/nlp-course/chapter7/4?fw=pt#fine-tuning-the-model
        training_args = Seq2SeqTrainingArguments(
            evaluation_strategy="epoch",
            output_dir=str(self.params.dirs.checkpoint),
            num_train_epochs=self.params.num_train_epochs,
            per_device_train_batch_size=self.params.per_device_batch_size,
            per_device_eval_batch_size=self.params.per_device_batch_size,
            learning_rate=2e-5,
            weight_decay=0.01,
            predict_with_generate=True,
            logging_dir=str(self.params.dirs.logging),
            report_to=["tensorboard"],
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset_dict["train"],
            eval_dataset=dataset_dict["test"],
            compute_metrics=compute_metrics,
        )

        trainer.train()
        trainer.save_model(str(self.params.dirs.model))

        metrics = trainer.evaluate(eval_dataset=dataset_dict["test"])
        trainer.save_metrics("all", metrics)
