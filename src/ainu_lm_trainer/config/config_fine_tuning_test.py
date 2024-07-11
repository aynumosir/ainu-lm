from .config_fine_tuning import FineTuningConfig


def test_fine_tuning_config_with_model_name() -> None:
    config = FineTuningConfig(base_model="base_model")

    assert config.base_model == "base_model"
    assert config.base_tokenizer == "base_model"


def test_fine_tuning_config_with_tokenizer_name() -> None:
    config = FineTuningConfig(base_model="base_model", base_tokenizer="base_tokenizer")

    assert config.base_model == "base_model"
    assert config.base_tokenizer == "base_tokenizer"
