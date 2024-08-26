from typing import Iterator

from datasets import Dataset, concatenate_datasets
from tokenizers import SentencePieceUnigramTokenizer

from ....config import DatasetConfig, WorkspaceConfig


class SentencepieceTokenizerTrainer:
    __dataset: Dataset
    __config_workspace: WorkspaceConfig

    def __init__(
        self, config_dataset: DatasetConfig, config_workspace: WorkspaceConfig
    ) -> None:
        self.__config_workspace = config_workspace

        dataset_dict = config_dataset.load()
        self.__dataset = concatenate_datasets(
            [dataset_dict["train"], dataset_dict["test"]]
        )

    # https://huggingface.co/docs/tokenizers/en/training_from_memory#using-the-datasets-library
    def __batch_iterator(self, batch_size: int = 1_000) -> Iterator[Iterator[str]]:
        for i in range(0, len(self.__dataset), batch_size):
            yield self.__dataset[i : i + batch_size]["text"]
            yield self.__dataset[i : i + batch_size]["translation"]

            # 方言名がちゃんとトークナイズされること
            for dialect in self.__dataset[i : i + batch_size]["dialect"]:
                if dialect:
                    yield dialect

    def __prepare(self) -> None:
        self.__config_workspace.model_dir.mkdir(parents=True, exist_ok=True)

    def train(self) -> SentencePieceUnigramTokenizer:
        self.__prepare()

        tokenizer = SentencePieceUnigramTokenizer()

        tokenizer.train_from_iterator(
            iterator=self.__batch_iterator(),
            unk_token="<unk>",
            special_tokens=[
                "<unk>",
                "<pad>",
                "</s>",
            ],
        )

        self.demo(tokenizer)
        tokenizer.save(str(self.__config_workspace.model_dir / "tokenizer.json"))

        return tokenizer

    def demo(self, tokenizer: SentencePieceUnigramTokenizer) -> None:
        # cSpell:disable
        text = "kanto or wa yaku sak no a=ranke p sinep ka isam 天から降ろされたもので役割が無いものは一つも無い。"
        # cSpell:enable

        encoded = tokenizer.encode(text).ids
        decoded = [tokenizer.decode([token_id]) for token_id in encoded]

        print(decoded)
