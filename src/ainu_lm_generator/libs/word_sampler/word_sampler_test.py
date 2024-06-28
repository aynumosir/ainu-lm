import random

from .word_sampler import WordSampler


def test_word_sampler() -> None:
    random.seed(42)
    word_sampler = WordSampler(["apple", "banana", "cherry"])
    assert word_sampler.sample() == "banana"
