import os
from pathlib import Path

from .bpe_trainer import BpeTrainer


def test_bpe_trainer() -> None:
    corpus = """
    hello
    world
    """

    with open("/tmp/corpus.txt", "w") as f:
        f.write(corpus)

    output_dir = Path("/tmp/")

    trainer = BpeTrainer(corpus_file=Path("/tmp/corpus.txt"), output_dir=output_dir)

    trainer.train()

    assert os.path.exists(output_dir.joinpath("tokenizer.json"))

    os.remove("/tmp/corpus.txt")
    os.remove("/tmp/tokenizer.json")
