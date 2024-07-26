from transformers import TrainingArguments

from .config_training import TrainingConfig


def test_config_training_with_none() -> None:
    config = TrainingConfig()

    assert config.num_train_epochs is None
    assert config.per_device_train_batch_size is None
    assert config.per_device_eval_batch_size is None
    assert config.gradient_accumulation_steps is None
    assert config.weight_decay is None
    assert config.learning_rate is None
    assert config.warmup_ratio is None
    assert config.push_to_hub is False


def test_config_training_with_values() -> None:
    config = TrainingConfig(
        per_device_eval_batch_size=16,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        warmup_ratio=0.1,
        gradient_accumulation_steps=2,
        learning_rate=3e-4,
        push_to_hub=True,
    )

    assert config.num_train_epochs == 3
    assert config.per_device_train_batch_size == 16
    assert config.per_device_eval_batch_size == 16
    assert config.gradient_accumulation_steps == 2
    assert config.weight_decay == 0.01
    assert config.learning_rate == 3e-4
    assert config.warmup_ratio == 0.1
    assert config.push_to_hub is True


def test_training_extend() -> None:
    args = TrainingArguments(
        output_dir=".",
        num_train_epochs=5,
        per_device_eval_batch_size=32,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=4,
        weight_decay=0.02,
        learning_rate=1e-4,
        warmup_ratio=0.2,
        hub_model_id="hub_model_id",
        push_to_hub=False,
    )

    config = TrainingConfig(
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=16,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        learning_rate=3e-4,
        warmup_ratio=0.1,
        hub_model_id="override_hub_model_id",
        push_to_hub=True,
    )

    extended_args = config.extend(args)

    assert extended_args.num_train_epochs == 3
    assert extended_args.gradient_accumulation_steps == 2
    assert extended_args.per_device_train_batch_size == 16
    assert extended_args.per_device_eval_batch_size == 16
    assert extended_args.weight_decay == 0.01
    assert extended_args.learning_rate == 3e-4
    assert extended_args.warmup_ratio == 0.1
    assert extended_args.hub_model_id == "override_hub_model_id"
    assert extended_args.push_to_hub is True
