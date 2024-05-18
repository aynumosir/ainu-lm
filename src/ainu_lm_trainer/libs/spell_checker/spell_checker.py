from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import MutableMapping, Optional

from ..corpus import Corpus
from .lcs import lcs
from .leven import leven


@dataclass
class SpellCheckEntry:
    word: str
    score: float


Cache = MutableMapping[str, list[SpellCheckEntry]]


class SpellChecker:
    __words: set[str]
    __cache: Optional[Cache]
    __top_k: int = 5

    def __init__(self, words: set[str], cache: Optional[Cache] = None) -> None:
        self.__words = words
        self.__cache = cache

    def check(self, w1: str) -> list[SpellCheckEntry]:
        if self.__cache is not None and w1 in self.__cache:
            return self.__cache[w1]

        entries: list[SpellCheckEntry] = [
            SpellCheckEntry(
                word=w2,
                score=self.__score(w1, w2, max_d=5, min_L=2),
            )
            for w2 in self.__words
            if w2 != w1
        ]

        entries = sorted(entries, key=lambda x: x.score, reverse=True)
        entries = entries[: self.__top_k]

        if self.__cache is not None:
            self.__cache[w1] = entries

        return entries

    def __score(self, w1: str, w2: str, max_d: int, min_L: int) -> float:
        d = leven(w1, w2)
        L = len(lcs(w1, w2))

        if d > max_d or L < min_L:
            return 0.0

        d_scaled = d / max_d
        l_scaled = L / min(len(w1), len(w2))

        score = (1 - d_scaled) + l_scaled / 2
        score = math.exp(score)

        return score

    @classmethod
    def from_corpus(
        cls,
        corpus: Corpus,
        min_frequency: int = 5,
        cache: Optional[Cache] = None,
    ) -> SpellChecker:
        usage = corpus.get_usages()
        usage = [
            u
            for u in usage
            if u.count >= min_frequency and re.match(r"[a-zA-Z]", u.word) is not None
        ]

        words = set(u.word for u in usage)
        return cls(words, cache)
