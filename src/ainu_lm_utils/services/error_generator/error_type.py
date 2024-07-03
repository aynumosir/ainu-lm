from __future__ import annotations

import enum
import random


class ErrorType(enum.Enum):
    REPLACE = 1
    DELETE = 2
    INSERT = 3
    SWAP = 4

    @staticmethod
    def random() -> ErrorType:
        return random.choices(list(ErrorType), k=1, weights=[0.7, 0.1, 0.1, 0.1])[0]
