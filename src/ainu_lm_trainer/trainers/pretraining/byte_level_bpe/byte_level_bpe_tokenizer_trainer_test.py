import os
from pathlib import Path

from datasets import Dataset, DatasetDict

from ....config import DatasetsConfigWithValue, WorkspaceConfig
from .byte_level_bpe_tokenizer_trainer import ByteLevelBpeTokenizerTrainer


def test_train() -> None:
    dataset = Dataset.from_dict(
        {
            "text": [
                "This is a sentence.",
                "This is another sentence.",
            ]
        }
    )
    dataset_dict = DatasetDict({"train": dataset, "test": dataset})

    model_dir = Path("/tmp/byte_level_tokenizer_test")

    trainer = ByteLevelBpeTokenizerTrainer(
        config_dataset=DatasetsConfigWithValue(dataset_dict),
        config_workspace=WorkspaceConfig(
            model_dir=model_dir,
        ),
    )
    trainer.train()

    assert os.path.exists(model_dir / "tokenizer.json")
