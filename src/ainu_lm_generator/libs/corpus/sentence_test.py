from .sentence import Sentence
from .word import Word


def test_iterating_words() -> None:
    sentence = Sentence([Word("Apple"), Word("banana")])

    assert list(sentence) == [Word("Apple"), Word("banana")]
