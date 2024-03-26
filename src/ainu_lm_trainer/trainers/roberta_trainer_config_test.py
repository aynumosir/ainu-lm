from pathlib import Path

from .roberta_trainer_config import RobertaTrainerConfig


def test_roberta_trainer_config() -> None:
    config = RobertaTrainerConfig(
        num_train_epochs=1,
        tokenizer_name_or_dir="tokenizer",
        model_dir=Path("model"),
        checkpoint_dir=Path("checkpoint"),
        logging_dir=Path("logging"),
    )

    assert config.num_train_epochs == 1
    assert config.tokenizer_name_or_dir == "tokenizer"
    assert str(config.model_dir) == "model"
    assert str(config.checkpoint_dir) == "checkpoint"
    assert str(config.logging_dir) == "logging"
    assert config.hypertune_enabled is False
