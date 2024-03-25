from pathlib import Path

from datasets import Dataset

from .roberta_trainer import RobertaTrainer
from .roberta_trainer_config import RobertaTrainerConfig


def test_compact_dataset() -> None:
    dataset = Dataset.from_dict(
        {
            "text": [
                "This is a test.",
                "This is another test.",
                "This is yet another test.",
            ]
        }
    )

    output_dir = Path("/tmp/ainu_lm_trainer_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    logging_dir = Path("/tmp/ainu_lm_trainer_test_logging")
    logging_dir.mkdir(parents=True, exist_ok=True)

    trainer = RobertaTrainer(
        dataset=dataset,
        config=RobertaTrainerConfig(
            num_train_epochs=1,
            tokenizer_name_or_dir="roberta-base",
            output_dir=output_dir,
            logging_dir=output_dir,
        ),
    )

    trainer.train()

    assert (output_dir / "config.json").exists()
