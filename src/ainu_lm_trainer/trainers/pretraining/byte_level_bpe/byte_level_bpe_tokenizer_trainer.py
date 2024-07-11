from typing import Iterator

from datasets import Dataset
from tokenizers import ByteLevelBPETokenizer
from tokenizers.normalizers import NFD, Lowercase, Sequence, StripAccents

from ....config import DatasetConfig, WorkspaceConfig


class ByteLevelBpeTokenizerTrainer:
    __dataset: Dataset
    __config_workspace: WorkspaceConfig

    def __init__(
        self, config_dataset: DatasetConfig, config_workspace: WorkspaceConfig
    ) -> None:
        self.__dataset = config_dataset.load()["train"]
        self.__config_workspace = config_workspace

    # https://huggingface.co/docs/tokenizers/en/training_from_memory#using-the-datasets-library
    def __batch_iterator(self, batch_size: int = 1_000) -> Iterator[Iterator[str]]:
        for i in range(0, len(self.__dataset), batch_size):
            yield self.__dataset[i : i + batch_size]["text"]

    def __prepare(self) -> None:
        self.__config_workspace.model_dir.mkdir(parents=True, exist_ok=True)

    def train(self) -> ByteLevelBPETokenizer:
        self.__prepare()

        tokenizer = ByteLevelBPETokenizer()

        # You can `lowercase=True` and `unicode_normalizer="nfd"` but unable to strip accents thus overriding the normalizer.
        # c.f. https://huggingface.co/docs/tokenizers/en/pipeline#normalization
        tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])

        tokenizer.train_from_iterator(
            iterator=self.__batch_iterator(),
            vocab_size=52_000,
            min_frequency=2,
            special_tokens=[
                "<s>",
                "<pad>",
                "</s>",
                "<unk>",
                "<mask>",
            ],
        )

        self.demo(tokenizer)
        tokenizer.save(str(self.__config_workspace.model_dir / "tokenizer.json"))

        return tokenizer

    def demo(self, tokenizer: ByteLevelBPETokenizer) -> None:
        # cSpell:disable
        text = "Kánto or wa yakú sak no a=ránke p sinép ka isám"
        # cSpell:enable

        print("=" * 80)
        print(f"text: {text}")
        print(f"encode(text): {tokenizer.encode(text).ids}")
        print(f"decode(text): {tokenizer.decode(tokenizer.encode(text).ids)}")
        print("=" * 80)
