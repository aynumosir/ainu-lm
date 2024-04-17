from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from datasets import Dataset, load_dataset


class TrainingDataset(ABC):
    @abstractmethod
    def get_dataset(self) -> Dataset:
        pass

    @abstractmethod
    def get_dataset_raw(self) -> Dataset:
        pass


@dataclass
class TrainingDatasetSource(TrainingDataset):
    name: str
    column_name: str = "text"
    split: str = "train"
    revision: Optional[str] = None

    def get_dataset(self) -> Dataset:
        dataset = load_dataset(self.name, split=self.split, revision=self.revision)
        return dataset.map(lambda example: {"text": example[self.column_name]})

    def get_dataset_raw(self) -> Dataset:
        return load_dataset(self.name, split=self.split, revision=self.revision)


@dataclass
class TrainingDatasetValue(TrainingDataset):
    dataset: Dataset

    def get_dataset(self) -> Dataset:
        return self.dataset

    def get_dataset_raw(self) -> Dataset:
        return self.dataset
