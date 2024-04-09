import torch
from transformers import (
    DataCollatorForLanguageModeling,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizerFast,
    Trainer,
    TrainingArguments,
)

from .roberta_trainer_params import RobertaTrainerParams


class RobertaTrainer:
    params: RobertaTrainerParams

    def __init__(self, params: RobertaTrainerParams) -> None:
        self.params = params

    def train(self) -> None:
        tokenizer = RobertaTokenizerFast.from_pretrained(str(self.params.tokenizer))

        config = RobertaConfig.from_pretrained("FacebookAI/roberta-base")

        model = RobertaForMaskedLM(config)
        model = model.to("cuda") if torch.cuda.is_available() else model

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=0.15
        )

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
