import os
from pathlib import Path

from datasets import Dataset

from ...config import DatasetsConfigWithValue, WorkspaceConfig
from .byte_level_bpe_tokenizer_trainer import ByteLevelBpeTokenizerTrainer


def test_train() -> None:
    dataset = Dataset.from_dict(
        {
            "sentence": [
                "This is a sentence.",
                "This is another sentence.",
            ]
        }
    )

    model_dir = Path("/tmp/byte_level_tokenizer_test")

    trainer = ByteLevelBpeTokenizerTrainer(
        dataset_config=DatasetsConfigWithValue(dataset),
        workspace_config=WorkspaceConfig(
            model_dir=model_dir,
        ),
    )
    trainer.train()

    assert os.path.exists(model_dir / "tokenizer.json")
