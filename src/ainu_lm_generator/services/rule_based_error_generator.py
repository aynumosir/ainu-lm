import multiprocessing
import os

from datasets import Dataset, interleave_datasets, load_dataset

from ..libs import Corpus, ErrorGenerator, SpellChecker, WordSampler


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


def generate_rule_based_errors(
    dataset_name: str, repeat: int, push_to_hub: bool
) -> None:
    dataset = load_dataset("aynumosir/ainu-corpora", split="train")
    dataset.filter(lambda example: len(example["sentence"]) > 0)

    error_generator = create_error_generator(dataset)

    if repeat > 1:
        dataset = interleave_datasets([dataset] * repeat)

    dataset = dataset.map(
        lambda example: {
            "text": error_generator(example["sentence"]),
            "target": example["sentence"],
        },
        num_proc=os.cpu_count(),
        remove_columns=dataset.column_names,
    )

    if push_to_hub:
        dataset.push_to_hub(dataset_name)
    else:
        dataset.save_to_disk(dataset_name)
