from datetime import datetime
from pathlib import Path


def get_pipeline_path(pipeline: str) -> Path:
    return Path(f"dist/ainu_{pipeline}_pipeline.yaml")


def get_template_path(region: str, pipeline: str) -> str:
    return f"https://{region}-kfp.pkg.dev/neetlab/kfp/ainu-{pipeline}-pipeline/latest"


def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S")
