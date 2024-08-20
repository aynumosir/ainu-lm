from argparse import ArgumentParser, Namespace

from ainu_utils import to_kana
from datasets import load_dataset

from ....services.normalizer.normalizer_ainu import (
    deduplicate_whitespace,
    remove_glottal_stop_before_affix,
    remove_linking_symbol,
    strip_accents,
)


def add_parser(parser: ArgumentParser) -> None:
    # fmt: off
    parser.add_argument("--input", type=str, default="aynumosir/ainu-corpora")
    parser.add_argument("--output", type=str, default="aynumosir/ainu-corpora-kana")
    # fmt: on


def sanitize(input: str) -> str:
    input = input.lower()
    input = strip_accents(input)
    input = remove_linking_symbol(input)
    input = remove_glottal_stop_before_affix(input)
    input = deduplicate_whitespace(input)
    return input


def main(args: Namespace) -> None:
    dataset_dict = load_dataset(args.input)
    dataset_dict = dataset_dict.filter(lambda example: len(example["text"]) > 0)
    dataset_dict = dataset_dict.map(
        lambda example: {
            "text": sanitize(example["text"]),
            "kana": to_kana(example["text"]),
        },
        remove_columns=dataset_dict["train"].column_names,
    )

    dataset_dict.push_to_hub(args.output)
