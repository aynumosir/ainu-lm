from google.cloud.aiplatform import PipelineJob
from kfp.compiler import Compiler

from ..pipelines import ainu_t5_gce_pipeline
from .get_timestamp import get_timestamp

if __name__ == "__main__":
    compiler = Compiler()
    compiler.compile(
        pipeline_func=ainu_t5_gce_pipeline,
        package_path="./dist/ainu_t5_gce_pipeline.yaml",
        pipeline_parameters={
            "project_id": "neetlab",
            "location": "us-central1",
            "train_image_uri": "us-central1-docker.pkg.dev/neetlab/docker/ainu-lm-trainer",
            "pipeline_staging": "gs://ainu-lm/staging",
            "tensorboard_id": "4565774810897973248",
            "service_account": "ainu-lm@neetlab.iam.gserviceaccount.com",
            "hf_model_repo": "aynumosir/t5-base-ainu-gce",
            "hf_dataset_repo": "aynumosir/ainu-corpora",
            "hf_secret_id": "aynumosir-hf-token",
            "github_repo": "aynumosir/ainu-lm",
            "github_secret_id": "aynumosir-github-token",
            "github_commit_sha": "5cd248434b1ea53f6fb63dcedec0f75fa6f73277",
            "push_to_hub": True,
        },
    )

    pipeline_job = PipelineJob(
        display_name="Ainu LM via Pull Request",
        template_path="./dist/ainu_t5_gce_pipeline.yaml",
        job_id=f"ainu-lm-t5-gce-{get_timestamp()}",
        pipeline_root="gs://ainu-lm/pipeline_root",
    )

    pipeline_job.submit(service_account="ainu-lm@neetlab.iam.gserviceaccount.com")
