import argparse
import os

from google.cloud.storage import Blob

from ..utils import get_path_from_uri


def get_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a language model")
    subparsers = parser.add_subparsers(dest="task", description="Model to train")

    """
    Subparser for the tokenizer
    """
    tokenizer_parser = subparsers.add_parser("tokenizer")
    tokenizer_parser.add_argument(
        "--output-dir",
        type=get_path_from_uri,
        help="Job directory. Use gs:/ to save to Google Cloud Storage",
        default=os.environ.get("AIP_MODEL_DIR"),
    )

    """
    Subparser for the language model
    """
    language_model_parser = subparsers.add_parser("language-model")
    language_model_parser.add_argument(
        "--hp-tune",
        default=False,
        help="Whether to use hyperparameter tuning",
    )
    language_model_parser.add_argument(
        "--num-train-epochs", type=int, help="Number of training epochs", default=10
    )
    language_model_parser.add_argument(
        "--tokenizer-dir",
        type=Blob.from_string,
        help="Tokenizer directory. Use gs:/ to load from Google Cloud Storage",
        required=True,
    )
    language_model_parser.add_argument(
        "--output-dir",
        type=get_path_from_uri,
        help="Job directory. Use gs:/ to save to Google Cloud Storage",
        default=os.environ.get("AIP_MODEL_DIR"),
    )
    language_model_parser.add_argument(
        "--logging-dir",
        type=get_path_from_uri,
        help="Logging directory. Use gs:/ to save to Google Cloud Storage",
        default=os.environ.get("AIP_TENSORBOARD_LOG_DIR"),
    )

    """
    Subparser for the cache
    """
    subparsers.add_parser("cache")

    return parser
