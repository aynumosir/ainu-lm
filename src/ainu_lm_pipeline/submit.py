import argparse
import os
from datetime import datetime

from google.cloud import aiplatform
from google.cloud.aiplatform.pipeline_jobs import PipelineJob


def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S")


PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")
TRAIN_IMAGE_URI = os.getenv("TRAIN_IMAGE_URI")
PIPELINE_ROOT = os.getenv("PIPELINE_ROOT")
PIPELINE_STAGING = os.getenv("PIPELINE_STAGING")
TENSORBOARD_NAME = os.getenv("TENSORBOARD_NAME")
SERVICE_ACCOUNT = os.getenv("SERVICE_ACCOUNT")


parser = argparse.ArgumentParser()
parser.add_argument("--commit-sha", type=str, required=True)
parser.add_argument("--no-cache", type=bool, default=False)


if __name__ == "__main__":
    aiplatform.init(project=PROJECT_ID, location=REGION)

    args = parser.parse_args()
    job_id = f"pipeline-ainu-lm-{get_timestamp()}"

    pipeline_params = {
        "project_id": PROJECT_ID,
        "location": REGION,
        "service_account": SERVICE_ACCOUNT,
        "tensorboard_name": TENSORBOARD_NAME,
        "train_image_uri": TRAIN_IMAGE_URI,
        "pipeline_job_id": job_id,
        "pipeline_staging": PIPELINE_STAGING,
        "source_repo_name": "github_aynumosir_ainu-lm",
        "source_commit_sha": args.commit_sha,
        "hf_repo": "aynumosir/roberta-ainu-base",
        "hf_secret_id": "aynumosir-hf-token",
    }

    pipeline_job = PipelineJob(
        display_name=f"Ainu LM Pipeline ({args.commit_sha})",
        template_path="./dist/ainu_lm_pipeline.json",
        job_id=job_id,
        pipeline_root=PIPELINE_ROOT,
        parameter_values=pipeline_params,
        enable_caching=not args.no_cache,
    )

    pipeline_job.run(sync=True)
