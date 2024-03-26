from pathlib import Path
from typing import Optional

from datasets import load_dataset

from ..trainers import RobertaTrainer, RobertaTrainerConfig


def language_model(
    output_dir: Path,
    logging_dir: Path,
    tokenizer_dir: Path,
    num_train_epochs: int,
    hypertune_enabled: Optional[bool] = None,
    tensorboard_id: Optional[str] = None,
    tensorboard_experiment_name: Optional[str] = None,
) -> None:
    dataset = load_dataset("aynumosir/ainu-corpora", split="data")
    dataset = dataset.map(lambda example: {"text": example["sentence"]})

    config = RobertaTrainerConfig(
        output_dir=output_dir,
        logging_dir=logging_dir,
        tokenizer_name_or_dir=tokenizer_dir,
        num_train_epochs=num_train_epochs,
        hypertune_enabled=hypertune_enabled,
    )

    trainer = RobertaTrainer(dataset, config=config)
    trainer.train()
