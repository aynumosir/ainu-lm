from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TrainingDirs:
    model: Path
    logging: Optional[Path] = None
    checkpoint: Optional[Path] = None
