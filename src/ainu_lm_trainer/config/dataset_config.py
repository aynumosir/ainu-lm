from abc import ABC
from dataclasses import dataclass
from typing import Optional

from datasets import Dataset, load_dataset


class DatasetConfig(ABC):
    def load(self) -> Dataset:
        raise NotImplementedError


@dataclass
class DatasetsConfigWithHuggingFaceHub(DatasetConfig):
    name: str
    split: Optional[str] = None
    revision: Optional[str] = None

    def load(self) -> Dataset:
        return load_dataset(self.name, revision=self.revision, split=self.split)


@dataclass
class DatasetsConfigWithValue(DatasetConfig):
    value: Dataset

    def load(self) -> Dataset:
        return self.value
