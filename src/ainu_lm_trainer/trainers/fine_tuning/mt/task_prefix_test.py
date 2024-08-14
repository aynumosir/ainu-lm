from ....config import TaskPrefixType
from .task_prefix import ain2ja

sentence = {
    "dialect": "静内",
    "pronoun": "first",
}

sentence_without_dialect = {
    "dialect": None,
    "pronoun": "first",
}


def test_ain2ja_none() -> None:
    assert ain2ja(sentence, TaskPrefixType.NONE) == "translate Ainu to Japanese: "


def test_ain2ja_dialect() -> None:
    assert (
        ain2ja(sentence, TaskPrefixType.DIALECT)
        == "translate Ainu (静内) to Japanese: "
    )


def test_ain2ja_pronoun() -> None:
    assert (
        ain2ja(sentence, TaskPrefixType.PRONOUN)
        == "translate Ainu (first) to Japanese: "
    )


def test_ain2ja_all() -> None:
    assert (
        ain2ja(sentence, TaskPrefixType.ALL)
        == "translate Ainu (静内, first) to Japanese: "
    )


def test_ain2ja_none_without_dialect() -> None:
    assert (
        ain2ja(sentence_without_dialect, TaskPrefixType.NONE)
        == "translate Ainu to Japanese: "
    )


def test_ja2ain_none() -> None:
    assert ain2ja(sentence, TaskPrefixType.NONE) == "translate Ainu to Japanese: "


def test_ja2ain_dialect() -> None:
    assert (
        ain2ja(sentence, TaskPrefixType.DIALECT)
        == "translate Ainu (静内) to Japanese: "
    )


def test_ja2ain_pronoun() -> None:
    assert (
        ain2ja(sentence, TaskPrefixType.PRONOUN)
        == "translate Ainu (first) to Japanese: "
    )


def test_ja2ain_all() -> None:
    assert (
        ain2ja(sentence, TaskPrefixType.ALL)
        == "translate Ainu (静内, first) to Japanese: "
    )


def test_ja2ain_none_without_dialect() -> None:
    assert (
        ain2ja(sentence_without_dialect, TaskPrefixType.NONE)
        == "translate Ainu to Japanese: "
    )
