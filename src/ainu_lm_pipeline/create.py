import argparse
import os
from pathlib import Path

from kfp.compiler import Compiler
from kfp.registry import RegistryClient

from ainu_lm_pipeline.pipelines import ainu_lm_pipeline


def get_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", type=str, default=os.getenv("PROJECT_ID"))
    parser.add_argument("--region", type=str, default=os.getenv("REGION"))
    parser.add_argument(
        "--train-image-uri",
        type=str,
        default=os.getenv("TRAIN_IMAGE_URI"),
        help="Docker image URI for training the model. e.g. gcr.io/my-project/my-image:latest",
    )
    parser.add_argument(
        "--pipeline-staging",
        type=str,
        default=os.getenv("PIPELINE_STAGING"),
        help="GCS path for pipeline staging. e.g. gs://my-bucket/pipeline/staging",
    )
    parser.add_argument(
        "--tensorboard-id",
        type=str,
        default=os.getenv("TENSORBOARD_ID"),
        help="ID of the TensorBoard instance. e.g. 4565774810897973248",
    )
    parser.add_argument(
        "--service-account",
        type=str,
        default=os.getenv("SERVICE_ACCOUNT"),
        help="Service account for the pipeline. e.g. my_account@my_project.iam.gserviceaccount.com",
    )
    parser.add_argument(
        "--hf-model-repo",
        type=str,
        help="Hugging Face model repository. e.g. aynumosir/roberta-base-ainu",
    )
    parser.add_argument(
        "--hf-dataset-repo",
        type=str,
        default=os.getenv("HF_DATASET_REPO"),
        help="Hugging Face dataset repository. e.g. aynumosir/ainu-corpora",
    )
    parser.add_argument(
        "--hf-secret-id",
        type=str,
        default=os.getenv("HF_SECRET_ID"),
        help="Secret ID for Hugging Face token. e.g. aynumosir-hf-token",
    )
    parser.add_argument(
        "--github-repo",
        type=str,
        default=os.getenv("GITHUB_REPO"),
        help="GitHub repository. e.g. aynumosir/ainu-lm",
    )
    parser.add_argument(
        "--github-secret-id",
        type=str,
        default=os.getenv("GITHUB_SECRET_ID"),
        help="Secret ID for GitHub token. e.g. aynumosir-github-token",
    )
    parser.add_argument(
        "--kfp-repo",
        type=str,
        default=os.getenv("KFP_REPO"),
        help="KFP repository. e.g. kfp",
    )
    return parser


if __name__ == "__main__":
    args = get_argument_parser().parse_args()
    pipeline_path = Path("./dist/ainu_lm_pipeline.yaml")
    os.makedirs(pipeline_path.parent, exist_ok=True)

    compiler = Compiler()
    compiler.compile(
        pipeline_func=ainu_lm_pipeline,
        package_path=str(pipeline_path),
        pipeline_parameters={
            "project_id": args.project_id,
            "location": args.region,
            "train_image_uri": args.train_image_uri,
            "pipeline_staging": args.pipeline_staging,
            "tensorboard_id": args.tensorboard_id,
            "service_account": args.service_account,
            "hf_model_repo": args.hf_model_repo,
            "hf_dataset_repo": args.hf_dataset_repo,
            "hf_secret_id": args.hf_secret_id,
            "github_repo": args.github_repo,
            "github_secret_id": args.github_secret_id,
        },
    )

    registry_client = RegistryClient(
        host=f"https://{args.region}-kfp.pkg.dev/{args.project_id}/{args.kfp_repo}"
    )

    template_name, version_name = registry_client.upload_pipeline(
        file_name=str(pipeline_path),
        tags=["latest"],
    )

    print("~" * 80)
    print(f"Pipeline template: {template_name}")
    print(f"Pipeline version: {version_name}")
    print("~" * 80)
