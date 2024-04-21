from pathlib import Path

from ..models import TrainingDatasetSource, TrainingDirs
from ..services import (
    MT5GECTrainer,
    MT5GECTrainerParams,
)


def mt5_gec(
    model_dir: Path,
    checkpoint_dir: Path,
    logging_dir: Path,
    num_train_epochs: int,
    per_device_batch_size: int,
    # tokenizer_dir: Path,
    dataset_revision: str,
) -> None:
    params = MT5GECTrainerParams(
        # tokenizer=tokenizer_dir,
        dirs=TrainingDirs(
            model=model_dir,
            checkpoint=checkpoint_dir,
            logging=logging_dir,
        ),
        dataset=TrainingDatasetSource(
            name="aynumosir/ainu-synthetic-learner-corpus",
            split="data",
            revision=dataset_revision,
            column_name="sentence",
        ),
        num_train_epochs=num_train_epochs,
        per_device_batch_size=per_device_batch_size,
    )

    trainer = MT5GECTrainer(params)
    trainer.train()
