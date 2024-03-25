from __future__ import annotations

from google.cloud import storage
from google.cloud.storage import Blob


class JobDir:
    """
    GCPに「ディレクトリ」の概念がないので作ったラッパー。
    """

    __blob: Blob

    def __init__(self, job_dir: str) -> None:
        self.__blob = Blob.from_string(job_dir, client=None)

    def resolve(self, name: str) -> JobDir:
        return JobDir(f"gs://{self.__blob.bucket.name}/{self.__blob.name}/{name}")

    def to_blob(self, client: storage.Client) -> Blob:
        blob = Blob.from_string(str(self), client=client)
        return blob

    def __str__(self) -> str:
        return f"gs://{self.__blob.bucket.name}/{self.__blob.name}"
