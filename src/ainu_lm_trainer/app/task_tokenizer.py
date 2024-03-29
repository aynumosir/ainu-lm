from pathlib import Path

from datasets import load_dataset

from ..trainers import ByteLevelBPETokenizerTrainer


def tokenizer(output_dir: Path, dataset_revision: str) -> None:
    dataset = load_dataset(
        "aynumosir/ainu-corpora", split="train", revision=dataset_revision
    )
    trainer = ByteLevelBPETokenizerTrainer(dataset, output_dir=output_dir)
    trainer.train()
