from __future__ import annotations

import unicodedata
from dataclasses import dataclass


@dataclass
class Word:
    value: str

    def __str__(self) -> str:
        return self.value

    def normalize(self) -> Word:
        value = self.value.lower().strip()
        value = "".join(
            c
            for c in unicodedata.normalize("NFKD", value)
            if not unicodedata.combining(c)
        )
        return Word(value)
