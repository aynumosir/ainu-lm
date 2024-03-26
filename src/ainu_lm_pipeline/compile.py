import os
from pathlib import Path

from kfp.compiler import Compiler
from kfp.registry import RegistryClient

from ainu_lm_pipeline.pipelines import ainu_lm_pipeline

PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")
TRAIN_IMAGE_URI = os.getenv("TRAIN_IMAGE_URI")
PIPELINE_ROOT = os.getenv("PIPELINE_ROOT")
PIPELINE_STAGING = os.getenv("PIPELINE_STAGING")
TENSORBOARD_NAME = os.getenv("TENSORBOARD_NAME")
SERVICE_ACCOUNT = os.getenv("SERVICE_ACCOUNT")


if __name__ == "__main__":
    pipeline_path = Path("./dist/ainu_lm_pipeline.yaml")

    pipeline_parameters = {
        "project_id": PROJECT_ID,
        "location": REGION,
        "service_account": SERVICE_ACCOUNT,
        "tensorboard_name": TENSORBOARD_NAME,
        "train_image_uri": TRAIN_IMAGE_URI,
        "pipeline_staging": PIPELINE_STAGING,
        "source_repo_name": "github_aynumosir_ainu-lm",
        "hf_model_repo": "aynumosir/roberta-base-ainu",
        "hf_dataset_repo": "aynumosir/ainu-corpora",
        "hf_secret_id": "aynumosir-hf-token",
        "github_repo": "aynumosir/ainu-lm",
        "github_secret_id": "aynumosir-github-token",
    }

    compiler = Compiler()

    compiler.compile(
        pipeline_func=ainu_lm_pipeline,
        package_path=str(pipeline_path),
        pipeline_parameters=pipeline_parameters,
    )

    registry_client = RegistryClient(host="https://us-central1-kfp.pkg.dev/neetlab/kfp")

    template_name, version_name = registry_client.upload_pipeline(
        file_name=str(pipeline_path),
        tags=["latest"],
    )

    print(f"Pipeline template: {template_name}")
    print(f"Pipeline version: {version_name}")
