from dataclasses import dataclass
from typing import Iterator

from .word import Word


@dataclass
class Sentence:
    words: list[Word]

    def __iter__(self) -> Iterator[Word]:
        return iter(self.words)
