from pathlib import Path


def get_path_from_uri(path: str | Path) -> Path:
    if isinstance(path, Path):
        return path

    if path.startswith("gs://"):
        return Path("/gcs/" + path[5:])
    return Path(path)
