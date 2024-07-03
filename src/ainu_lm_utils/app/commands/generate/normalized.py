from argparse import ArgumentParser, Namespace

from datasets import load_dataset

from ....services import normalizer


def add_parser(parser: ArgumentParser) -> None:
    # fmt: off
    parser.add_argument("--input", type=str, default="aynumosir/ainu-corpora")
    parser.add_argument("--output", type=str, default="aynumosir/ainu-corpora-normalized")
    # fmt: on


def main(args: Namespace) -> None:
    dataset_dict = load_dataset(args.input)
    dataset_dict = dataset_dict.map(normalizer.normalize)

    # 表記方法が違うので除外
    dataset_dict = dataset_dict.filter(
        lambda example: example["title"] != "アイヌ語音声資料"
    )

    dataset_dict.push_to_hub(args.output)
