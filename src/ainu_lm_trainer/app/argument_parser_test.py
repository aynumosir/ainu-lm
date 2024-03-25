from .argument_parser import get_argument_parser


def test_parsing_tokenizer_training() -> None:
    parser = get_argument_parser()
    args = parser.parse_args(["tokenizer", "--job-dir=gs://test/job_dir"])
    assert args.task == "tokenizer"
    assert str(args.job_dir) == "gs://test/job_dir"


def test_parsing_language_model_training() -> None:
    parser = get_argument_parser()
    args = parser.parse_args(
        [
            "language-model",
            "--hp-tune=True",
            "--num-train-epochs=20",
            "--tokenizer-dir=gs://test/tokenizer",
            "--job-dir=gs://test/job_dir",
        ]
    )
    assert args.task == "language-model"
    assert args.hp_tune == "True"
    assert args.num_train_epochs == 20

    assert args.tokenizer_dir.bucket.name == "test"
    assert args.tokenizer_dir.name == "tokenizer"

    assert str(args.job_dir) == "gs://test/job_dir"


def test_parsing_language_model_with_tensorboard() -> None:
    parser = get_argument_parser()
    args = parser.parse_args(
        [
            "language-model",
            "--hp-tune=True",
            "--num-train-epochs=20",
            "--tokenizer-dir=gs://test/tokenizer",
            "--job-dir=gs://test/job_dir",
            "--tensorboard-id=123",
            "--tensorboard-experiment-name=exp",
        ]
    )
    assert args.task == "language-model"
    assert args.hp_tune == "True"
    assert args.num_train_epochs == 20

    assert args.tokenizer_dir.bucket.name == "test"
    assert args.tokenizer_dir.name == "tokenizer"

    assert str(args.job_dir) == "gs://test/job_dir"

    assert args.tensorboard_id == "123"
    assert args.tensorboard_experiment_name == "exp"
