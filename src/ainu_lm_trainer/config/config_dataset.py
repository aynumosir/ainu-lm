from abc import ABC
from typing import Optional

from datasets import DatasetDict, load_dataset

TRAIN_SPLIT = "train"
TEST_SPLIT = "test"


def validate(
    dataset_dict: DatasetDict,
) -> None:
    splits = dataset_dict.keys()

    if TRAIN_SPLIT not in splits or TEST_SPLIT not in splits:
        raise ValueError("Dataset must have 'train' and 'test' splits")


class DatasetConfig(ABC):
    def load(self) -> DatasetDict:
        raise NotImplementedError


class DatasetsConfigWithHuggingFaceHub(DatasetConfig):
    def __init__(
        self,
        name: str,
        revision: Optional[str] = None,
    ) -> None:
        self.name = name
        self.revision = revision

    def load(self) -> DatasetDict:
        dataset_dict = load_dataset(self.name, revision=self.revision)
        validate(dataset_dict)
        return dataset_dict


class DatasetsConfigWithValue(DatasetConfig):
    value: DatasetDict

    def __init__(self, value: DatasetDict) -> None:
        validate(value)
        self.value = value

    def load(self) -> DatasetDict:
        return self.value
