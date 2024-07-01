from __future__ import annotations

from dataclasses import dataclass

from ainu_utils import segment
from datasets import Dataset

from .sentence import Sentence
from .word import Word


@dataclass
class Usage:
    word: str
    count: int


@dataclass
class Corpus:
    sentences: list[Sentence]

    def get_usages(self) -> list[Usage]:
        usages: dict = {}

        for sentence in self.sentences:
            for word in sentence:
                s = str(word.normalize())

                if s in usages:
                    usages[s] += 1
                else:
                    usages[s] = 1

        return [Usage(word, count) for word, count in usages.items()]

    @classmethod
    def from_dataset(cls, dataset: Dataset, column_name: str = "text") -> Corpus:
        sentences: list[Sentence] = []

        for example in dataset[column_name]:
            words = [Word(w) for w in segment(example, keep_whitespace=False)]
            sentence = Sentence(words)
            sentences.append(sentence)

        return cls(sentences)
