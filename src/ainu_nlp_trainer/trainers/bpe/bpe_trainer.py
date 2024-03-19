import os
from pathlib import Path

from tokenizers import Regex, Tokenizer, normalizers, pre_tokenizers, trainers
from tokenizers.models import BPE


class BpeTrainer:
    corpus_file: Path
    output_dir: Path
    tokenizer: Tokenizer

    def __init__(
        self,
        corpus_file: Path,
        output_dir: Path = Path("/tmp/ainu_nlp_trainer/sentencepiece/"),
    ) -> None:
        self.corpus_file = corpus_file
        self.output_dir = output_dir
        self.tokenizer = Tokenizer(BPE())

    def train(self) -> Tokenizer:
        os.makedirs(self.output_dir, exist_ok=True)

        self.tokenizer.normalizer = normalizers.Sequence(
            [
                normalizers.Nmt(),
                normalizers.NFKC(),
                normalizers.Replace(Regex(" {2,}"), " "),
                normalizers.Lowercase(),
            ]
        )

        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        trainer = trainers.BpeTrainer(
            vocab_size=32000,
            min_frequency=2,
            show_progress=True,
        )

        self.tokenizer.train(
            [str(self.corpus_file)],
            trainer=trainer,
        )

        self.tokenizer.save(str(self.output_dir / "tokenizer.json"))

        return self.tokenizer
