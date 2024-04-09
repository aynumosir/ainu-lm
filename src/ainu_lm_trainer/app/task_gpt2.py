from pathlib import Path

from ..models import TrainingDatasetSource, TrainingDirs
from ..services import (
    GPT2Trainer,
    GPT2TrainerParams,
)


def gpt2(
    model_dir: Path,
    checkpoint_dir: Path,
    logging_dir: Path,
    tokenizer_dir: Path,
    num_train_epochs: int,
    dataset_revision: str,
) -> None:
    params = GPT2TrainerParams(
        dirs=TrainingDirs(
            model=model_dir,
            checkpoint=checkpoint_dir,
            logging=logging_dir,
        ),
        dataset=TrainingDatasetSource(
            name="aynumosir/ainu-corpora",
            split="train",
            revision=dataset_revision,
            column_name="sentence",
        ),
        tokenizer=tokenizer_dir,
        num_train_epochs=num_train_epochs,
    )

    trainer = GPT2Trainer(params)
    trainer.train()
