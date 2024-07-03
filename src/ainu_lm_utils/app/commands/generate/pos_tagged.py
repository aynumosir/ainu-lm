from argparse import ArgumentParser, Namespace

from datasets import load_dataset
from transformers import AutoTokenizer, pipeline


def add_parser(parser: ArgumentParser) -> None:
    # fmt: off
    parser.add_argument("--input", type=str, default="aynumosir/ainu-corpora-normalized")
    parser.add_argument("--output", type=str, default="aynumosir/ainu-corpora-pos-tagged")
    parser.add_argument("--pos-tagger", type=str, default="aynumosir/roberta-base-ainu-pos")
    # fmt: on


def main(args: Namespace) -> None:
    dataset_dict = load_dataset(args.input)

    # debug
    dataset_dict["train"] = dataset_dict["train"].select(range(1000))

    tokenizer = AutoTokenizer.from_pretrained(args.pos_tagger, model_max_length=128)
    pos_tagger = pipeline(
        "ner",
        model=args.pos_tagger,
        tokenizer=tokenizer,
    )

    def pos_tag(example: dict) -> dict:
        entities = pos_tagger(example["text"], aggregation_strategy="simple")

        return {
            "words": [entity["word"] for entity in entities],
            "pos": [entity["entity_group"] for entity in entities],
        }

    dataset_dict = dataset_dict.map(pos_tag, remove_columns=["translation"])
    dataset_dict.push_to_hub(args.output)
