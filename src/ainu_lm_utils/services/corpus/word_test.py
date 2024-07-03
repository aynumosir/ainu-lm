from .word import Word


def test_normalizing_word() -> None:
    word = Word("Kamúy")
    assert str(word.normalize()) == "kamuy"
