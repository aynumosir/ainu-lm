from google.cloud.aiplatform import PipelineJob
from kfp.compiler import Compiler

from ..pipelines import ainu_mt5_gec_pipeline
from .get_timestamp import get_timestamp

if __name__ == "__main__":
    compiler = Compiler()
    compiler.compile(
        pipeline_func=ainu_mt5_gec_pipeline,
        package_path="./dist/ainu_mt5_gec_pipeline.yaml",
        pipeline_parameters={
            "project_id": "neetlab",
            "location": "us-central1",
            "train_image_uri": "us-central1-docker.pkg.dev/neetlab/docker/ainu-lm-trainer",
            "pipeline_staging": "gs://ainu-lm/staging",
            "tensorboard_id": "4565774810897973248",
            "service_account": "ainu-lm@neetlab.iam.gserviceaccount.com",
            "hf_model_repo": "aynumosir/mt5-base-ainu-gec",
            "hf_dataset_repo": "aynumosir/ainu-corpora",
            "hf_secret_id": "aynumosir-hf-token",
            "github_repo": "aynumosir/ainu-lm",
            "github_secret_id": "aynumosir-github-token",
            "github_commit_sha": "76472405177f7579a50f3c0aad05958f38aca72f",
            "push_to_hub": True,
        },
    )

    pipeline_job = PipelineJob(
        display_name="Ainu LM via Pull Request",
        template_path="./dist/ainu_mt5_gec_pipeline.yaml",
        job_id=f"ainu-lm-mt5-gec-{get_timestamp()}",
        pipeline_root="gs://ainu-lm/pipeline_root",
    )

    pipeline_job.submit(service_account="ainu-lm@neetlab.iam.gserviceaccount.com")
