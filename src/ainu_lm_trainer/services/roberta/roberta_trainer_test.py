from pathlib import Path

from datasets import Dataset

from ...config import (
    DatasetsConfigWithValue,
    FineTuningConfig,
    TrainingConfig,
    WorkspaceConfig,
)
from .roberta_trainer import RobertaTrainer


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

    model_dir = Path("/tmp/ainu_lm_trainer_test")
    model_dir.mkdir(parents=True, exist_ok=True)

    logging_dir = Path("/tmp/ainu_lm_trainer_test_logging")
    logging_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path("/tmp/ainu_lm_trainer_test_checkpoint")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trainer = RobertaTrainer(
        dataset_config=DatasetsConfigWithValue(dataset),
        fine_tuning_config=FineTuningConfig(tokenizer="roberta-base"),
        training_config=TrainingConfig(
            num_train_epochs=1,
        ),
        workspace_config=WorkspaceConfig(
            model_dir=model_dir,
            logging_dir=logging_dir,
            checkpoint_dir=checkpoint_dir,
        ),
    )

    trainer.train()

    assert (model_dir / "config.json").exists()
    assert (checkpoint_dir / "all_results.json").exists()
