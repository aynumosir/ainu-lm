from pathlib import Path
from typing import Iterator

from datasets import Dataset
from tokenizers import ByteLevelBPETokenizer
from tokenizers.normalizers import NFD, Lowercase, Sequence, StripAccents


class ByteLevelBPETokenizerTrainer:
    dataset: Dataset
    output_dir: Path

    def __init__(
        self, dataset: Dataset, output_dir: Path = Path("models/tokenizer")
    ) -> None:
        self.dataset = dataset
        self.output_dir = output_dir

    # https://huggingface.co/docs/tokenizers/en/training_from_memory#using-the-datasets-library
    def batch_iterator(self, batch_size: int = 1_000) -> Iterator[Iterator[str]]:
        for i in range(0, len(self.dataset), batch_size):
            yield self.dataset[i : i + batch_size]["sentence"]

    def prepare(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self) -> ByteLevelBPETokenizer:
        self.prepare()

        tokenizer = ByteLevelBPETokenizer()

        # You can `lowercase=True` and `unicode_normalizer="nfd"` but unable to set strip accents
        # https://huggingface.co/docs/tokenizers/en/pipeline#normalization
        tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])

        tokenizer.train_from_iterator(
            iterator=self.batch_iterator(),
            vocab_size=52_000,
            min_frequency=2,
            special_tokens=[
                "<s>",
                "<pad>",
                "</s>",
                "<unk>",
                "<mask>",
            ],
        )

        tokenizer.save(str(self.output_dir / "tokenizer.json"))
