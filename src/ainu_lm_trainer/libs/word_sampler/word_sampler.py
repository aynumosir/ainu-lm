from __future__ import annotations

import random

from ..corpus import Corpus


class WordSampler:
    __items: list
    __probabilities: list

    def __init__(self, words: list[str]) -> None:
        word_to_count = {}

        for word in words:
            if word not in word_to_count:
                word_to_count[word] = 1
            else:
                word_to_count[word] += 1

        self.__items = list(word_to_count.keys())
        self.__probabilities = [count / len(words) for count in word_to_count.values()]

    def sample(self) -> str:
        return random.choices(self.__items, weights=self.__probabilities, k=1)[0]

    @classmethod
    def from_corpus(cls, corpus: Corpus) -> WordSampler:
        return cls(
            [
                str(word)
                for word in [
                    word.normalize()
                    for sentence in corpus.sentences
                    for word in sentence
                ]
            ]
        )
