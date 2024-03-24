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
    def __init__(self, dataset: Dataset, output_dir: Path = Path("models/lm")) -> None:
        self.dataset = dataset
        self.output_dir = output_dir

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
            "./models/tokenizer",
            max_length=512,
            padding="max_length",
            truncation=True,
        )

        # 一応、vocab_size が合っているか確認
        assert tokenizer.vocab_size == config.vocab_size

        model = RobertaForMaskedLM(config=config)
        model = model.to("cuda") if torch.cuda.is_available() else model

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            overwrite_output_dir=True,
            num_train_epochs=10,
            per_device_train_batch_size=64,
            save_steps=10_000,
            save_total_limit=2,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=0.15
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=self.dataset,
        )

        trainer.train()

        return trainer
