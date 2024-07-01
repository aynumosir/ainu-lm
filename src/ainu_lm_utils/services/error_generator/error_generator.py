import math
import random

from ...utils.sllist import Sllist, SllistNode
from ..spell_checker import SpellChecker
from ..word_sampler import WordSampler
from .error_type import ErrorType
from .tagged_word import TaggedWord
from .well_known_replacements import WellKnownReplacements


class ErrorGenerator:
    word_sampler: WordSampler
    spell_checker: SpellChecker
    well_known_replacements = WellKnownReplacements()

    def __init__(self, word_sampler: WordSampler, spell_checker: SpellChecker) -> None:
        self.word_sampler = word_sampler
        self.spell_checker = spell_checker

    def __call__(self, tagged_words: list[TaggedWord]) -> list[TaggedWord]:
        return self.err(tagged_words)

    def err(self, _tagged_words: list[TaggedWord]) -> list[TaggedWord]:
        tagged_words = Sllist.from_list(_tagged_words)
        tagged_word_nodes = list(tagged_words.iter_nodes())

        error_probability = min(max(random.normalvariate(0.15, 0.2), 0), 1)
        error_count = math.floor(len(tagged_word_nodes) * error_probability)
        error_words = random.sample(tagged_word_nodes, k=error_count)

        for error_word in error_words:
            error_type = ErrorType.random()

            if error_type == ErrorType.REPLACE:
                self.__replace(error_word)
            elif error_type == ErrorType.INSERT:
                self.__insert(error_word)
            elif error_type == ErrorType.DELETE:
                self.__delete(error_word)
            elif error_type == ErrorType.SWAP:
                self.__swap(error_word)

        return list(tagged_words.iter_values())

    def __replace(self, node: SllistNode[TaggedWord]) -> None:
        replacement = self.well_known_replacements.get(node.value)
        if replacement is not None:
            node.value = TaggedWord(replacement, "UNK")
            return

        entries = self.spell_checker.suggest(node.value.word)

        if len(entries) == 0:
            return

        samples = random.choices(
            entries, k=1, weights=[entry.score for entry in entries]
        )
        node.value = TaggedWord(samples[0].word, "UNK")

    def __insert(self, node: SllistNode[TaggedWord]) -> None:
        value = self.word_sampler.sample()
        node.insert_after(SllistNode(TaggedWord(value, "UNK")))

    def __delete(self, node: SllistNode[TaggedWord]) -> None:
        node.remove()

    def __swap(self, node: SllistNode[TaggedWord]) -> None:
        if node.next is None:
            return
        else:
            node.value, node.next.value = node.next.value, node.value
