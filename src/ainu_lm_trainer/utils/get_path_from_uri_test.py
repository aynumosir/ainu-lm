from pathlib import Path

from .get_path_from_uri import get_path_from_uri


def test_with_local_path() -> None:
    path = get_path_from_uri("/some/local/path")
    assert path == Path("/some/local/path")


def test_with_gs_path() -> None:
    path = get_path_from_uri("gs://foo/bar")
    assert path == Path("/gcs/foo/bar")
