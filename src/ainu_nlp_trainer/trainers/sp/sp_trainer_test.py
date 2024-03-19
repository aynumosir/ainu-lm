import os
from pathlib import Path

from .sp_trainer import SentencePieceTrainer


def test_sentencepiece_trainer() -> None:
    corpus = """
    hello
    world
    """

    with open("/tmp/corpus.txt", "w") as f:
        f.write(corpus)

    trainer = SentencePieceTrainer(
        corpus_file=Path("/tmp/corpus.txt"),
        output_dir=Path("/tmp/"),
        vocab_size=11,
        min_log_level=11,
    )

    result = trainer.train()

    assert os.path.exists(result.model_path)
    assert os.path.exists(result.vocab_path)

    os.remove(result.model_path)
