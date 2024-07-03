from argparse import ArgumentParser, Namespace

import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import pipeline


def add_parser(parser: ArgumentParser) -> None:
    # fmt: off
    parser.add_argument("--input", type=str, default="aynumosir/ainu-corpora-normalized")
    parser.add_argument("--output", type=str, default="aynumosir/ainu-corpora-round-trip-translations")
    parser.add_argument("--model", type=str, default="aynumosir/mt5-small-ainu")
    parser.add_argument("--num-return-sequences", type=int, default=20, help="Number of back-translated sequences to generate")
    # fmt: on


def get_task_prefix(example: dict) -> str:
    # マージ前に戻す
    # if example["dialect"] is not None:
    #     return (
    #         f"translate Japanese to Ainu ({example['dialect']}, {example['pronoun']}): "
    #     )
    # else:
    #     return f"translate Japanese to Ainu (沙流, {example['pronoun']}): "
    return "translate Japanese to Ainu: "


def main(args: Namespace) -> None:
    dataset = load_dataset(args.input, split="train")

    # debug
    dataset = dataset.select(range(10))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    translate = pipeline("translation", model=args.model, device=device)

    rows = []

    for example in tqdm(dataset):
        task_prefix = get_task_prefix(example)

        translations = translate(
            task_prefix + example["translation"],
            max_length=128,
            do_sample=True,
            num_return_sequences=args.num_return_sequences,
        )

        for translation in translations:
            new_row = example.copy()

            # もっとマシな書き方があるかも
            del new_row["translation"]
            new_row["target"] = example["text"]
            new_row["text"] = translation["translation_text"]

            rows.append(new_row)

    dataset = Dataset.from_list(rows)
    dataset.push_to_hub(args.output)
