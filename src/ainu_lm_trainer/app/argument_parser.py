import argparse
import os

from google.cloud.storage import Blob

from ..models.job_dir import JobDir


def get_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a language model")
    subparsers = parser.add_subparsers(dest="task", description="Model to train")

    """
    Subparser for the tokenizer
    """
    tokenizer_parser = subparsers.add_parser("tokenizer")
    tokenizer_parser.add_argument(
        "--job-dir",
        type=JobDir,
        help="Job directory. Use gs:/ to save to Google Cloud Storage",
        default=os.environ.get("AIP_MODEL_DIR"),
    )

    """
    Subparser for the language model
    """
    language_model_parser = subparsers.add_parser("language_model")
    language_model_parser.add_argument(
        "--hp-tune",
        default=False,
        help="Whether to use hyperparameter tuning",
    )
    language_model_parser.add_argument(
        "--model-name",
        type=str,
        help="Model name to train (e.g. roberta-base-ainu)",
        default=os.environ.get("MODEL_NAME"),
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
        "--job-dir",
        type=JobDir,
        help="Job directory. Use gs:/ to save to Google Cloud Storage",
        default=os.environ.get("AIP_MODEL_DIR"),
    )

    """
    Subparser for the cache
    """
    subparsers.add_parser("cache")

    return parser
