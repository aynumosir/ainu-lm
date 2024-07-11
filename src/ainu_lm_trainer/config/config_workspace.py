from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class WorkspaceConfig:
    model_dir: Path
    checkpoint_dir: Optional[Path] = None
    logging_dir: Optional[Path] = None
