from .config_mt_experiments import (
    MtExperimentsConfig,
    PronounType,
    TaskPrefixType,
)


def test_mt_experiments_config() -> None:
    config = MtExperimentsConfig()
    assert config.hyperparameter_tuning is False
    assert config.task_prefix == TaskPrefixType.ALL
    assert config.include_dialect is None
    assert config.include_pronoun is None

    config = MtExperimentsConfig(
        hyperparameter_tuning=True,
        task_prefix=TaskPrefixType.DIALECT,
        include_dialect="dialect",
        include_pronoun=PronounType.FIRST,
    )
    assert config.hyperparameter_tuning is True
    assert config.task_prefix == TaskPrefixType.DIALECT
    assert config.include_dialect == "dialect"
    assert config.include_pronoun == PronounType.FIRST


def test_pronoun_type() -> None:
    assert PronounType.from_str("first") == PronounType.FIRST
    assert PronounType.from_str("fourth") == PronounType.FOURTH


def test_task_prefix_type() -> None:
    assert TaskPrefixType.from_str("none") == TaskPrefixType.NONE
    assert TaskPrefixType.from_str("dialect") == TaskPrefixType.DIALECT
    assert TaskPrefixType.from_str("pronoun") == TaskPrefixType.PRONOUN
    assert TaskPrefixType.from_str("all") == TaskPrefixType.ALL
