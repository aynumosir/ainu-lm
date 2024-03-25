from pathlib import Path

from .roberta_trainer_config import RobertaTrainerConfig


def test_roberta_trainer_config() -> None:
    config = RobertaTrainerConfig(
        num_train_epochs=1,
        tokenizer_name_or_dir="tokenizer",
        output_dir=Path("output"),
        logging_dir=Path("logging"),
    )

    assert config.num_train_epochs == 1
    assert config.tokenizer_name_or_dir == "tokenizer"
    assert str(config.output_dir) == "output"
    assert config.hypertune_enabled is False
