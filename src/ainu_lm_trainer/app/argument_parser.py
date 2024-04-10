import argparse
import os

from ..utils import get_path_from_uri


def get_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a language model")
    subparsers = parser.add_subparsers(dest="task", description="Model to train")

    """
    Subparser for the Byte-Level BPE
    """
    byte_level_bpe = subparsers.add_parser("byte-level-bpe")
    byte_level_bpe.add_argument(
        "--output-dir",
        type=get_path_from_uri,
        help="Job directory. Use gs:/ to save to Google Cloud Storage",
        default=os.environ.get("AIP_MODEL_DIR"),
    )
    byte_level_bpe.add_argument(
        "--dataset-revision",
        type=str,
        help="Dataset version e.g. v1",
    )

    """
    Subparser for the Sentencepiece
    """
    sentencepiece = subparsers.add_parser("sentencepiece")
    sentencepiece.add_argument(
        "--output-dir",
        type=get_path_from_uri,
        help="Job directory. Use gs:/ to save to Google Cloud Storage",
        default=os.environ.get("AIP_MODEL_DIR"),
    )
    sentencepiece.add_argument(
        "--dataset-revision",
        type=str,
        help="Dataset version e.g. v1",
    )

    """
    Subparser for the RoBERTa
    """
    roberta = subparsers.add_parser("roberta")
    roberta.add_argument(
        "--num-train-epochs", type=int, help="Number of training epochs", default=10
    )
    roberta.add_argument(
        "--tokenizer-dir",
        type=get_path_from_uri,
        help="Tokenizer directory. Use gs:/ to load from Google Cloud Storage",
        required=True,
    )
    roberta.add_argument(
        "--model-dir",
        type=get_path_from_uri,
        help="Job directory. Use gs:/ to save to Google Cloud Storage",
        default=os.environ.get("AIP_MODEL_DIR"),
    )
    roberta.add_argument(
        "--checkpoint-dir",
        type=get_path_from_uri,
        help="Checkpoint directory. Use gs:/ to save to Google Cloud Storage",
        default=os.environ.get("AIP_CHECKPOINT_DIR"),
    )
    roberta.add_argument(
        "--logging-dir",
        type=get_path_from_uri,
        help="Logging directory. Use gs:/ to save to Google Cloud Storage",
        default=os.environ.get("AIP_TENSORBOARD_LOG_DIR"),
    )
    roberta.add_argument(
        "--dataset-revision",
        type=str,
        help="Dataset version e.g. v1",
    )
    roberta.add_argument(
        "--per-device-batch-size",
        type=int,
        help="Per device batch size",
    )

    """
    Subparser for the GPT2
    """
    gpt2 = subparsers.add_parser("gpt2")
    gpt2.add_argument(
        "--num-train-epochs", type=int, help="Number of training epochs", default=10
    )
    gpt2.add_argument(
        "--tokenizer-dir",
        type=get_path_from_uri,
        help="Tokenizer directory. Use gs:/ to load from Google Cloud Storage",
        required=True,
    )
    gpt2.add_argument(
        "--model-dir",
        type=get_path_from_uri,
        help="Job directory. Use gs:/ to save to Google Cloud Storage",
        default=os.environ.get("AIP_MODEL_DIR"),
    )
    gpt2.add_argument(
        "--checkpoint-dir",
        type=get_path_from_uri,
        help="Checkpoint directory. Use gs:/ to save to Google Cloud Storage",
        default=os.environ.get("AIP_CHECKPOINT_DIR"),
    )
    gpt2.add_argument(
        "--logging-dir",
        type=get_path_from_uri,
        help="Logging directory. Use gs:/ to save to Google Cloud Storage",
        default=os.environ.get("AIP_TENSORBOARD_LOG_DIR"),
    )
    gpt2.add_argument(
        "--dataset-revision",
        type=str,
        help="Dataset version e.g. v1",
    )

    """
    Subparser for the T5
    """
    t5 = subparsers.add_parser("t5")
    t5.add_argument(
        "--num-train-epochs", type=int, help="Number of training epochs", default=10
    )
    t5.add_argument(
        "--tokenizer-dir",
        type=get_path_from_uri,
        help="Tokenizer directory. Use gs:/ to load from Google Cloud Storage",
        required=True,
    )
    t5.add_argument(
        "--model-dir",
        type=get_path_from_uri,
        help="Job directory. Use gs:/ to save to Google Cloud Storage",
        default=os.environ.get("AIP_MODEL_DIR"),
    )
    t5.add_argument(
        "--checkpoint-dir",
        type=get_path_from_uri,
        help="Checkpoint directory. Use gs:/ to save to Google Cloud Storage",
        default=os.environ.get("AIP_CHECKPOINT_DIR"),
    )
    t5.add_argument(
        "--logging-dir",
        type=get_path_from_uri,
        help="Logging directory. Use gs:/ to save to Google Cloud Storage",
        default=os.environ.get("AIP_TENSORBOARD_LOG_DIR"),
    )
    t5.add_argument(
        "--dataset-revision",
        type=str,
        help="Dataset version e.g. v1",
    )
    t5.add_argument(
        "--per-device-batch-size",
        type=int,
        help="Per device batch size",
    )

    return parser
