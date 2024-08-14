from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TaskPrefixType(Enum):
    NONE = "none"
    DIALECT = "dialect"
    PRONOUN = "pronoun"
    ALL = "all"

    @staticmethod
    def from_str(value: str) -> TaskPrefixType:
        if value == "none":
            return TaskPrefixType.NONE
        if value == "dialect":
            return TaskPrefixType.DIALECT
        if value == "pronoun":
            return TaskPrefixType.PRONOUN
        if value == "all":
            return TaskPrefixType.ALL
        raise ValueError(f"Unknown TaskPrefixType: {value}")


class PronounType(Enum):
    FIRST = "first"
    FOURTH = "fourth"

    @staticmethod
    def from_str(value: str) -> PronounType:
        if value == "first":
            return PronounType.FIRST
        if value == "fourth":
            return PronounType.FOURTH
        raise ValueError(f"Unknown PronounType: {value}")


@dataclass
class MtExperimentsConfig:
    # Is hyperparameter tuning enabled?
    hyperparameter_tuning: bool = False
    # Task prefix type. Available values are "none" "dialect" "pronoun" "both".
    task_prefix: TaskPrefixType = TaskPrefixType.ALL
    # dialects to include
    include_dialect: Optional[str] = None
    # pronouns to include
    include_pronoun: Optional[PronounType] = None
