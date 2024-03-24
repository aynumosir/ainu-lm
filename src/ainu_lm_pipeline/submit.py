import argparse
import os

from google.cloud import aiplatform
from google.cloud.aiplatform.pipeline_jobs import PipelineJob

from . import config as cfg

parser = argparse.ArgumentParser()
parser.add_argument("--commit-sha", type=str, required=True)


if __name__ == "__main__":
    aiplatform.init(project=cfg.PROJECT_ID, location=cfg.REGION)

    args = parser.parse_args()
    job_id = f"pipeline-ainu-lm-{args.commit_sha}"

    pipeline_params = {
        "pipeline_job_id": job_id,
        "pipeline_root": cfg.PIPELINE_ROOT,
        "source_repo_name": "github_aynumosir_ainu-lm",
        "source_commit_sha": args.commit_sha,
        "tensorboard_id": os.environ.get("TENSORBOARD_ID"),
        "hf_repo": "aynumosir/roberta-ainu-base",
        "hf_token": os.environ.get("HF_TOKEN"),
    }

    pipeline_job = PipelineJob(
        display_name=f"pytorch-{cfg.APP_NAME}",
        template_path="./pipelines/ainu_lm_pipeline.json",
        job_id=job_id,
        pipeline_root=cfg.PIPELINE_ROOT,
        parameter_values=pipeline_params,
        enable_caching=True,
    )

    pipeline_job.run(sync=True)
