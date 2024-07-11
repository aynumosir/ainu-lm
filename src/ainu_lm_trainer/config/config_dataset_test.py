from datasets import Dataset, DatasetDict

from .config_dataset import DatasetsConfigWithValue

# def test_datasets_config_with_hugging_face_hub_load() -> None:
#     config = DatasetsConfigWithHuggingFaceHub(name="squad")
#     dataset = config.load()
#     assert dataset is not None


# def test_datasets_config_with_hugging_face_hub_load_revision() -> None:
#     config = DatasetsConfigWithHuggingFaceHub(
#         name="squad",
#         revision="1.0",
#     )
#     dataset = config.load()
#     assert dataset is not None


def test_datasets_config_with_value_load() -> None:
    train = Dataset.from_dict({"value": "value"})
    test = Dataset.from_dict({"value": "value"})
    dataset_dict = DatasetDict({"train": train, "test": test})

    config = DatasetsConfigWithValue(value=dataset_dict)
    dataset = config.load()
    assert dataset == dataset_dict
