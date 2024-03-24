import os
from dataclasses import dataclass
from pathlib import Path

import sentencepiece as spm


@dataclass
class SentencePieceTrainerResult:
    model_path: Path
    vocab_path: Path


class SentencePieceTrainer:
    # https://github.com/google/sentencepiece?tab=readme-ov-file#usage-instructions
    vocab_size: int
    character_coverage: float
    min_log_level: int

    corpus_file: Path
    output_dir: Path

    def __init__(
        self,
        corpus_file: Path,
        output_dir: Path = Path("/tmp/ainu_nlp_trainer/sentencepiece/"),
        vocab_size: int = 16000,
        character_coverage: float = 1.0,
        min_log_level: int = 0,
    ) -> None:
        self.corpus_file = corpus_file
        self.output_dir = output_dir
        self.vocab_size = vocab_size
        self.character_coverage = character_coverage
        self.min_log_level = min_log_level

    def train(self) -> SentencePieceTrainerResult:
        os.makedirs(self.output_dir, exist_ok=True)

        spm.SentencePieceTrainer.Train(
            input=self.corpus_file,
            model_prefix=self.output_dir.joinpath("tokenizer"),
            vocab_size=self.vocab_size,
            character_coverage=self.character_coverage,
            minloglevel=self.min_log_level,
        )

        return SentencePieceTrainerResult(
            model_path=self.output_dir.joinpath("tokenizer.model"),
            vocab_path=self.output_dir.joinpath("tokenizer.vocab"),
        )
