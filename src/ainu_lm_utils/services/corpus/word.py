from __future__ import annotations

import unicodedata
from dataclasses import dataclass


@dataclass
class Word:
    value: str

    def __str__(self) -> str:
        return self.value

    def normalize(self) -> Word:
        text = (
            "".join(
                char
                for char in unicodedata.normalize("NFKD", self.value)
                if not unicodedata.combining(char)
            )
            .lower()
            .strip()
        )

        return Word(text)
