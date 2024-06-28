from __future__ import annotations

import math
import random
import re
from typing import overload

from ainu_utils import segment
from pyllist import sllist, sllistnode

from ..spell_checker import SpellChecker
from ..word_sampler import WordSampler
from .error_type import ErrorType


def stringify(words: list[str]) -> str:
    text = "".join(words)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"=\s", "=", text)
    return text.strip()


def can_introduce_error(word: str) -> bool:
    is_not_empty = word.strip() != ""
    is_not_affix = "=" not in word
    is_alphabet = re.fullmatch(r"[a-zA-Z]+", word) is not None
    return is_not_empty and is_not_affix and is_alphabet


class ErrorGenerator:
    __word_sampler: WordSampler
    __spell_checker: SpellChecker

    def __init__(self, word_sampler: WordSampler, spell_checker: SpellChecker) -> None:
        self.__word_sampler = word_sampler
        self.__spell_checker = spell_checker

    @overload
    def __call__(self, sentences: list[str]) -> list[str]: ...
    @overload
    def __call__(self, sentences: str) -> str: ...
    def __call__(self, sentences: str | list[str]) -> str | list[str]:
        if isinstance(sentences, list):
            return [self.error(sentence) for sentence in sentences]
        else:
            return self.error(sentences)

    def error(self, sentence: str) -> str:
        words = sllist(segment(sentence, keep_whitespace=True))

        # TODO: このへんリファクタしたい
        valid_words = [
            error_word
            for error_word in words.iternodes()
            if can_introduce_error(error_word.value)
        ]

        if len(valid_words) == 0:
            return sentence

        error_probability = min(max(random.normalvariate(0.15, 0.2), 0), 1)
        error_count = math.floor(len(valid_words) * error_probability)
        error_words = random.sample(valid_words, k=error_count)

        for word in error_words:
            error_type = ErrorType.random()

            if error_type == ErrorType.REPLACE:
                self.__replace(word)
            elif error_type == ErrorType.INSERT:
                self.__insert(word, words)
            elif error_type == ErrorType.DELETE:
                self.__delete(word, words)
            elif error_type == ErrorType.SWAP:
                self.__swap(word)

        return stringify(words)

    def __replace(self, word: sllistnode, use_most_probable: bool = True) -> None:
        entries = self.__spell_checker.check(word.value)

        if len(entries) == 0:
            return

        if use_most_probable:
            entry = entries[0]
            word.value = entry.word
        else:
            samples = random.choices(
                entries, k=1, weights=[entry.score for entry in entries]
            )
            word.value = samples[0].word

    def __insert(self, word: sllistnode, owner: sllist) -> None:
        new_word = self.__word_sampler.sample()
        owner.insertafter(owner.insertafter(word, " "), new_word)

    def __delete(self, word: sllistnode, owner: sllist) -> None:
        owner.remove(word)

    def __swap(self, word: sllistnode) -> None:
        next_valid_word: sllistnode

        if word.next is not None and word.next.value.strip() != "":
            next_valid_word = word.next
        elif (
            word.next is not None
            and word.next.next is not None
            and word.next.next.value.strip() != ""
        ):
            next_valid_word = word.next.next
        else:
            next_valid_word = None

        if next_valid_word is not None:
            (word.value, next_valid_word.value) = (next_valid_word.value, word.value)
        else:
            pass
