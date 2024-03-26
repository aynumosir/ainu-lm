from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class RobertaTrainerConfig:
    model_dir: Path
    checkpoint_dir: Path
    logging_dir: Path

    num_train_epochs: int
    tokenizer_name_or_dir: Path | str
    hypertune_enabled: Optional[bool] = False
