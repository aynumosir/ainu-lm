from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class RobertaTrainerConfig:
    num_train_epochs: int
    tokenizer_name_or_dir: Path | str
    output_dir: Path

    hypertune_enabled: Optional[bool] = False

    tensorboard_id: Optional[str] = None
    tensorboard_experiment_name: Optional[str] = None

    @property
    def tensorboard_enabled(self) -> bool:
        return (
            self.tensorboard_id is not None
            and self.tensorboard_experiment_name is not None
        )
