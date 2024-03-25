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
        # FacebookAI/roberta-base よりも hidden_layers が少し小さい。エスペラントの記事を参考にした。
        # https://colab.research.google.com/github/huggingface/blog/blob/main/notebooks/01_how_to_train.ipynb
        roberta_config = RobertaConfig(
            vocab_size=52_000,
            max_position_embeddings=514,
            num_attention_heads=12,
            num_hidden_layers=6,
            type_vocab_size=1,
        )

        tokenizer = RobertaTokenizerFast.from_pretrained(
            str(self.config.tokenizer_name_or_dir)
        )

        model = RobertaForMaskedLM(config=roberta_config)
        model = model.to("cuda") if torch.cuda.is_available() else model

        training_args = TrainingArguments(
            output_dir=str(self.config.output_dir),
            overwrite_output_dir=True,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=16,
            save_steps=10_000,
            save_total_limit=2,
            logging_dir=str(self.config.logging_dir),
            report_to=["tensorboard"],
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=0.15
        )

        train_dataset = self.dataset.map(
            lambda examples: tokenizer(
                examples["text"],
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ),
            batched=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
        )

        if self.config.hypertune_enabled:
            trainer.add_callback(HPTuneCallback("loss", "eval_loss"))

        trainer.train()
        trainer.save_model(self.config.output_dir)
