from pathlib import Path

import torch
from datasets import Dataset
from google.cloud import aiplatform
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
    logging_dir: Path

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
        self.logging_dir = Path("./logs")

    def train(self) -> None:
        # FacebookAI/roberta-base よりも hidden_layers が少し小さい。エスペラントの記事を参考にした。
        # https://colab.research.google.com/github/huggingface/blog/blob/main/notebooks/01_how_to_train.ipynb
        config = RobertaConfig(
            vocab_size=52_000,
            max_position_embeddings=514,
            num_attention_heads=12,
            num_hidden_layers=6,
            type_vocab_size=1,
        )

        tokenizer = RobertaTokenizerFast.from_pretrained(
            str(self.config.tokenizer_name_or_dir),
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        if config.vocab_size != tokenizer.vocab_size:
            print(
                f"config.vocab_size ({config.vocab_size}) != tokenizer.vocab_size ({tokenizer.vocab_size})"
            )

        model = RobertaForMaskedLM(config=config)
        model = model.to("cuda") if torch.cuda.is_available() else model

        training_args = TrainingArguments(
            output_dir=str(self.config.output_dir),
            overwrite_output_dir=True,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=64,
            save_steps=10_000,
            save_total_limit=2,
            logging_dir=self.logging_dir,
            report_to=["tensorboard"] if self.config.tensorboard_enabled else [],
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=0.15
        )

        train_dataset = self.dataset.map(
            lambda examples: tokenizer(
                examples["text"], padding="max_length", truncation=True
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

        if self.config.tensorboard_enabled:
            aiplatform.start_upload_tb_log(
                tensorboard_id=config.tensorboard_id,
                tensorboard_experiment_name=config.tensorboard_experiment_name,
                logdir=str(self.logging_dir),
                run_name_prefix="roberta-base-ainu",
            )

        trainer.train()

        if self.config.tensorboard_enabled:
            aiplatform.end_upload_tb_log()

        trainer.save_model(self.config.output_dir)
