import torch
from datasets import Dataset
from transformers import (
    DataCollatorForLanguageModeling,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizerFast,
    Trainer,
    TrainingArguments,
)

from .roberta_trainer_callback import HPTuneCallback
from .roberta_trainer_config import RobertaTrainerConfig


class RobertaTrainer:
    dataset: Dataset
    config: RobertaTrainerConfig

    def __init__(
        self,
        dataset: Dataset,
        config: RobertaTrainerConfig,
    ) -> None:
        if "text" not in dataset.column_names:
            raise ValueError('The dataset must have a column named "text"')
        else:
            self.dataset = dataset
        self.config = config

    def train(self) -> None:
        roberta_config = RobertaConfig.from_pretrained("FacebookAI/roberta-base")

        tokenizer = RobertaTokenizerFast.from_pretrained(
            str(self.config.tokenizer_name_or_dir)
        )

        model = RobertaForMaskedLM(config=roberta_config)
        model = model.to("cuda") if torch.cuda.is_available() else model

        training_args = TrainingArguments(
            evaluation_strategy="epoch",
            output_dir=str(self.config.checkpoint_dir),
            overwrite_output_dir=True,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            logging_dir=str(self.config.logging_dir),
            report_to=["tensorboard"],
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=0.15
        )

        dataset = self.dataset.map(
            lambda examples: tokenizer(
                examples["text"],
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ),
            batched=True,
        )
        dataset_dict = dataset.train_test_split(test_size=0.1)

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset_dict["train"],
            eval_dataset=dataset_dict["test"],
        )

        if self.config.hypertune_enabled:
            trainer.add_callback(HPTuneCallback("loss", "eval_loss"))

        trainer.train()

        metrics = trainer.evaluate(eval_dataset=dataset_dict["test"])
        trainer.save_metrics("all", metrics)

        trainer.save_model(str(self.config.model_dir))
