from __future__ import annotations

from typing import Optional

from google.cloud import storage
from google.cloud.storage import Blob


class JobDir:
    """
    GCPに「ディレクトリ」の概念がないので作ったラッパー。
    """

    __blob: Blob

    def __init__(
        self, artifact_path: str, client: Optional[storage.Client] = None
    ) -> None:
        self.__blob = Blob.from_string(artifact_path, client=client)

    def resolve(self, name: str) -> JobDir:
        return JobDir(
            f"gs://{self.__blob.bucket.name}/{self.__blob.name}/{name}",
            client=self.__blob.client,
        )

    @property
    def blob(self) -> Blob:
        return self.__blob

    def __str__(self) -> str:
        return f"gs://{self.__blob.bucket.name}/{self.__blob.name}"
