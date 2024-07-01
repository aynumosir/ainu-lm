import multiprocessing
import os
from argparse import ArgumentParser, Namespace

from datasets import Dataset, interleave_datasets, load_dataset

from ....services import Corpus, ErrorGenerator, SpellChecker, TaggedWord, WordSampler


def add_parser(parser: ArgumentParser) -> None:
    # fmt: off
    parser.add_argument("--input", type=str, default="aynumosir/ainu-corpora-pos-tagged")
    parser.add_argument("--output", type=str, default="aynumosir/ainu-corpora-rule-based-errors")
    parser.add_argument("--repeat", type=int, default=1, help="Number of times to repeat the dataset")
    # fmt: on


def create_error_generator(dataset: Dataset) -> ErrorGenerator:
    corpus = Corpus.from_dataset(dataset)
    word_sampler = WordSampler.from_corpus(corpus)
    spell_checker_cache = multiprocessing.Manager().dict()
    spell_checker = SpellChecker.from_corpus(corpus, cache=spell_checker_cache)
    error_generator = ErrorGenerator(
        word_sampler=word_sampler,
        spell_checker=spell_checker,
    )
    return error_generator


def stringify_tagged_words(tagged_words: list[TaggedWord]) -> str:
    return "".join([str(tagged_word) for tagged_word in tagged_words]).strip()


def main(args: Namespace) -> None:
    dataset_dict = load_dataset(args.input)
    dataset_dict.filter(lambda example: len(example["text"]) > 0)
    dataset = interleave_datasets([dataset_dict["train"]] * args.repeat)

    error_generator = create_error_generator(dataset)

    def process(example: dict) -> dict:
        tagged_words = [
            TaggedWord(word=word, pos=pos)
            for word, pos in zip(example["words"], example["pos"])
        ]

        confused_words = error_generator(tagged_words)

        return {
            "text": stringify_tagged_words(tagged_words),
            "target": stringify_tagged_words(confused_words),
        }

    dataset = dataset.map(
        process, num_proc=os.cpu_count(), remove_columns=["words", "pos"]
    )

    dataset.push_to_hub(args.output)
