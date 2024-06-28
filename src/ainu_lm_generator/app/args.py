import os
from argparse import ArgumentParser


def get_parser() -> ArgumentParser:
    common = ArgumentParser(add_help=False)
    common.add_argument("--push-to-hub", type=bool, default=False)

    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="task")

    # Rule-based error generator
    rule_based_parser = subparsers.add_parser("rule-based", parents=[common])
    rule_based_parser.add_argument(
        "--dataset-name", type=str, default="aynumosir/ainu-gec-rule-based"
    )
    rule_based_parser.add_argument(
        "--repeat", type=int, default=1, help="Number of times to repeat the dataset"
    )

    # Back-translation
    back_translation_parser = subparsers.add_parser(
        "back-translation", parents=[common]
    )
    back_translation_parser.add_argument(
        "--dataset-name", type=str, default="aynumosir/ainu-gec-back-translation"
    )
    back_translation_parser.add_argument(
        "--inference-endpoint-url",
        type=str,
        default=os.getenv("INFERENCE_ENDPOINT_URL"),
        help="HuggingFace Inference Endpoint URL",
    )
    back_translation_parser.add_argument(
        "--semaphore-count",
        type=int,
        default=5,
        help="Number of concurrent requests to make",
    )
    back_translation_parser.add_argument(
        "--batch-size", type=int, default=64, help="Number of examples per batch"
    )
    back_translation_parser.add_argument(
        "--num-return-sequences",
        type=int,
        default=20,
        help="Number of back-translated sequences to generate",
    )

    return parser
