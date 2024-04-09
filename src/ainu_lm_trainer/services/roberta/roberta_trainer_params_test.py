from pathlib import Path

from ...models import TrainingDatasetSource, TrainingDirs
from .roberta_trainer_params import (
    RobertaTrainerParams,
)


def test_roberta_trainer_params() -> None:
    params = RobertaTrainerParams(
        num_train_epochs=1,
        tokenizer="tokenizer",
        dirs=TrainingDirs(
            model=Path("model"),
            checkpoint=Path("checkpoint"),
            logging=Path("logging"),
        ),
        dataset=TrainingDatasetSource(
            name="dataset",
            column_name="text",
        ),
    )

    assert params.num_train_epochs == 1
    assert params.tokenizer == "tokenizer"
    assert params.per_device_batch_size == 32
    assert params.context_length == 128

    assert str(params.dirs.model) == "model"
    assert str(params.dirs.checkpoint) == "checkpoint"
    assert str(params.dirs.logging) == "logging"
