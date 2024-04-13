from .argument_parser import get_argument_parser


def test_parsing_byte_level_bpe_training() -> None:
    parser = get_argument_parser()
    args = parser.parse_args(["byte-level-bpe", "--output-dir=gs://test/output_dir"])
    assert args.task == "byte-level-bpe"
    assert str(args.output_dir) == "/gcs/test/output_dir"


def test_parsing_roberta_training() -> None:
    parser = get_argument_parser()
    args = parser.parse_args(
        [
            "roberta",
            "--num-train-epochs=20",
            "--tokenizer-dir=gs://test/tokenizer",
            "--dataset-revision=v1",
            "--model-dir=gs://test/model_dir",
            "--logging-dir=gs://test/logging_dir",
            "--checkpoint-dir=gs://test/checkpoint_dir",
        ]
    )
    assert args.task == "roberta"
    assert args.num_train_epochs == 20
    assert args.dataset_revision == "v1"

    assert str(args.tokenizer_dir) == "/gcs/test/tokenizer"
    assert str(args.model_dir) == "/gcs/test/model_dir"
    assert str(args.logging_dir) == "/gcs/test/logging_dir"
    assert str(args.checkpoint_dir) == "/gcs/test/checkpoint_dir"


def test_parsing_roberta_with_local_disk() -> None:
    parser = get_argument_parser()
    args = parser.parse_args(
        [
            "roberta",
            "--num-train-epochs=20",
            "--tokenizer-dir=/model/tokenizer",
            "--dataset-revision=v1",
            "--model-dir=/model/model_dir",
            "--logging-dir=/model/logging_dir",
            "--checkpoint-dir=/model/checkpoint_dir",
        ]
    )
    assert args.task == "roberta"
    assert args.num_train_epochs == 20
    assert args.dataset_revision == "v1"

    assert str(args.tokenizer_dir) == "/model/tokenizer"
    assert str(args.model_dir) == "/model/model_dir"
    assert str(args.logging_dir) == "/model/logging_dir"
    assert str(args.checkpoint_dir) == "/model/checkpoint_dir"


def test_parsing_t5_training() -> None:
    parser = get_argument_parser()
    args = parser.parse_args(
        [
            "t5",
            "--num-train-epochs=20",
            "--per-device-batch-size=128",
            "--tokenizer-dir=gs://test/tokenizer",
            "--dataset-revision=v1",
            "--model-dir=gs://test/model_dir",
            "--logging-dir=gs://test/logging_dir",
            "--checkpoint-dir=gs://test/checkpoint_dir",
        ]
    )
    assert args.task == "t5"
    assert args.num_train_epochs == 20
    assert args.per_device_batch_size == 128
    assert args.dataset_revision == "v1"

    assert str(args.tokenizer_dir) == "/gcs/test/tokenizer"
    assert str(args.model_dir) == "/gcs/test/model_dir"
    assert str(args.logging_dir) == "/gcs/test/logging_dir"
    assert str(args.checkpoint_dir) == "/gcs/test/checkpoint_dir"
