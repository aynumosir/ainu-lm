from .argument_parser import get_argument_parser


def test_parsing_tokenizer_training() -> None:
    parser = get_argument_parser()
    args = parser.parse_args(["tokenizer", "--output-dir=gs://test/output_dir"])
    assert args.task == "tokenizer"
    assert str(args.output_dir) == "/gcs/test/output_dir"


def test_parsing_language_model_training() -> None:
    parser = get_argument_parser()
    args = parser.parse_args(
        [
            "language-model",
            "--hp-tune=True",
            "--num-train-epochs=20",
            "--tokenizer-dir=gs://test/tokenizer",
            "--output-dir=gs://test/output_dir",
            "--logging-dir=gs://test/logging_dir",
        ]
    )
    assert args.task == "language-model"
    assert args.hp_tune == "True"
    assert args.num_train_epochs == 20

    assert args.tokenizer_dir.bucket.name == "test"
    assert args.tokenizer_dir.name == "tokenizer"

    assert str(args.output_dir) == "/gcs/test/output_dir"
    assert str(args.logging_dir) == "/gcs/test/logging_dir"


def test_parsing_language_model_with_tensorboard() -> None:
    parser = get_argument_parser()
    args = parser.parse_args(
        [
            "language-model",
            "--hp-tune=True",
            "--num-train-epochs=20",
            "--tokenizer-dir=gs://test/tokenizer",
            "--output-dir=gs://test/output_dir",
            "--logging-dir=gs://test/logging_dir",
        ]
    )
    assert args.task == "language-model"
    assert args.hp_tune == "True"
    assert args.num_train_epochs == 20

    assert args.tokenizer_dir.bucket.name == "test"
    assert args.tokenizer_dir.name == "tokenizer"

    assert str(args.output_dir) == "/gcs/test/output_dir"
    assert str(args.logging_dir) == "/gcs/test/logging_dir"
