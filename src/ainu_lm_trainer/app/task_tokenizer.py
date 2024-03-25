from pathlib import Path

from datasets import load_dataset

from ..trainers import ByteLevelBPETokenizerTrainer


def tokenizer(output_dir: Path) -> None:
    dataset = load_dataset("aynumosir/ainu-corpora", split="data")
    trainer = ByteLevelBPETokenizerTrainer(dataset, output_dir=output_dir)
    trainer.train()
