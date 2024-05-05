from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class FineTuningConfig:
    tokenizer: Path | str
    model: Optional[Path | str] = None
