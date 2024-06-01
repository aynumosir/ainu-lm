from .corpus import Corpus, Usage
from .sentence import Sentence
from .word import Word


def test_making_word_set() -> None:
    corpus = Corpus(
        [
            Sentence([Word("Apple"), Word("banana")]),
            Sentence([Word("Banana"), Word("cherry")]),
        ]
    )

    assert corpus.get_usages() == [
        Usage("apple", 1),
        Usage("banana", 2),
        Usage("cherry", 1),
    ]
