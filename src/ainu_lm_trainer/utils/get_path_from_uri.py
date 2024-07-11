from pathlib import Path


def get_path_str_from_uri(path: str) -> str:
    if path.startswith("gs://"):
        return "/gcs/" + path[5:]
    else:
        return path


def get_path_from_uri(path: str) -> Path:
    return Path(get_path_str_from_uri(path))
