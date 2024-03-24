from pathlib import Path

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


class RobertaTrainer:
    dataset: Dataset
    output_dir: Path
    tokenizer_name_or_dir: Path | str

    def __init__(
        self, dataset: Dataset, tokenizer_name_or_dir: Path | str, output_dir: Path
    ) -> None:
        self.output_dir = output_dir
        self.tokenizer_name_or_dir = tokenizer_name_or_dir

        if "text" not in dataset.column_names:
            raise ValueError('The dataset must have a column named "text"')
        else:
            self.dataset = dataset

    def train(self, num_train_epochs: int = 10) -> None:
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
            str(self.tokenizer_name_or_dir),
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
            output_dir=str(self.output_dir),
            overwrite_output_dir=True,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=64,
            save_steps=10_000,
            save_total_limit=2,
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

        trainer.train()
        trainer.save_model(self.output_dir)
