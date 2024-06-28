# fmt: off
import os
from argparse import ArgumentParser

from ..utils import get_path_from_uri

# Base model arguments
base_model_parser = ArgumentParser(add_help=False)
base_model_parser.add_argument("--base-model", type=get_path_from_uri)
base_model_parser.add_argument("--base-tokenizer", type=get_path_from_uri)

# Dataset arguments
dataset_parser = ArgumentParser(add_help=False)
dataset_parser.add_argument("--dataset-name", type=str)
dataset_parser.add_argument("--dataset-revision", type=str)
dataset_parser.add_argument("--dataset-split", type=str)

# Training arguments
training_parser = ArgumentParser(add_help=False)
training_parser.add_argument("--per-device-train-batch-size", type=int)
training_parser.add_argument("--per-device-eval-batch-size", type=int)
training_parser.add_argument("--num-train-epochs", type=int)
training_parser.add_argument("--weight-decay", type=float)
training_parser.add_argument("--learning-rate", type=float)
training_parser.add_argument("--push-to-hub", type=bool)

# Workspace arguments
workspace_parser = ArgumentParser(add_help=False)
workspace_parser.add_argument("--model-dir", type=get_path_from_uri, default=os.environ.get("AIP_MODEL_DIR"))
workspace_parser.add_argument("--checkpoint-dir", type=get_path_from_uri, default=os.environ.get("AIP_CHECKPOINT_DIR"))
workspace_parser.add_argument("--logging-dir", type=get_path_from_uri, default=os.environ.get("AIP_TENSORBOARD_LOG_DIR"))


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    subparser = parser.add_subparsers(dest="task")

    parents = [
        base_model_parser,
        dataset_parser,
        training_parser,
        workspace_parser,
    ]

    subparser.add_parser("roberta", parents=parents)
    subparser.add_parser("roberta-pos", parents=parents)
    subparser.add_parser("mt5", parents=parents)
    subparser.add_parser("mt5-gec", parents=parents)
    subparser.add_parser("mt5-affix", parents=parents)
    subparser.add_parser("gpt2", parents=parents)
    subparser.add_parser("byte-level-bpe", parents=parents)

    return parser
