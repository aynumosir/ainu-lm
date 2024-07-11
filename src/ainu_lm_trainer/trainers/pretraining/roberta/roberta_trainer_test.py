from pathlib import Path

from datasets import Dataset, DatasetDict

from ....config import (
    DatasetsConfigWithValue,
    TrainingConfig,
    WorkspaceConfig,
)
from .roberta_trainer import train


def test_compact_dataset() -> None:
    dataset = Dataset.from_dict(
        {
            "text": [
                "this is a 1st test sentence",
                "this is a 2nd test sentence",
                "this is a 3rd test sentence",
                "this is a 4th test sentence",
                "this is a 5th test sentence",
                "this is a 6th test sentence",
                "this is a 7th test sentence",
                "this is a 8th test sentence",
                "this is a 9th test sentence",
                "this is a 10th test sentence",
            ]
        }
    )
    dataset_dict = DatasetDict({"train": dataset, "test": dataset})

    model_dir = Path("/tmp/ainu_lm_trainer_test")
    model_dir.mkdir(parents=True, exist_ok=True)

    logging_dir = Path("/tmp/ainu_lm_trainer_test_logging")
    logging_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path("/tmp/ainu_lm_trainer_test_checkpoint")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train(
        tokenizer_name="FacebookAI/roberta-base",
        config_dataset=DatasetsConfigWithValue(dataset_dict),
        config_training=TrainingConfig(
            num_train_epochs=1,
        ),
        config_workspace=WorkspaceConfig(
            model_dir=model_dir,
            logging_dir=logging_dir,
            checkpoint_dir=checkpoint_dir,
        ),
    )

    assert (model_dir / "config.json").exists()
