from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...models import TrainingDataset, TrainingDirs


@dataclass
class RobertaTrainerParams:
    dirs: TrainingDirs
    dataset: TrainingDataset

    tokenizer: Path | str

    num_train_epochs: int
    per_device_batch_size: int = 32
    context_length: int = 128
