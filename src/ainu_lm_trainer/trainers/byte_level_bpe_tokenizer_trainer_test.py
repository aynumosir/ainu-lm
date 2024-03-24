import os
from pathlib import Path

from datasets import Dataset

from .byte_level_bpe_tokenizer_trainer import ByteLevelBPETokenizerTrainer


def test_train() -> None:
    dataset = Dataset.from_dict(
        {
            "sentence": [
                "This is a sentence.",
                "This is another sentence.",
            ]
        }
    )

    trainer = ByteLevelBPETokenizerTrainer(
        dataset=dataset, output_dir=Path("/tmp/byte_level_tokenizer_test")
    )
    trainer.train()

    assert os.path.exists(trainer.output_dir / "tokenizer.json")
