from typing import Optional


class FineTuningConfig:
    __base_model: str
    __base_tokenizer: Optional[str]

    def __init__(
        self,
        base_model: str,
        base_tokenizer: Optional[str] = None,
    ) -> None:
        self.__base_model = base_model
        self.__base_tokenizer = base_tokenizer

    @property
    def base_model(self) -> str:
        return self.__base_model

    @property
    def base_tokenizer(self) -> str:
        return self.__base_tokenizer or self.__base_model
