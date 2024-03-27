import argparse
import os
from pathlib import Path

from google.cloud import aiplatform
from google.cloud.aiplatform import PipelineJob

from .config import pipeline_path
from .get_timestamp import get_timestamp


def get_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", type=str, default=os.getenv("PROJECT_ID"))
    parser.add_argument("--region", type=str, default=os.getenv("REGION"))
    parser.add_argument("--pipeline-root", type=str, default=os.getenv("PIPELINE_ROOT"))
    parser.add_argument(
        "--service-account",
        type=str,
        default=os.getenv("SERVICE_ACCOUNT"),
    )
    parser.add_argument(
        "--template-path",
        type=Path,
        default=pipeline_path,
    )
    parser.add_argument(
        "--github-commit-sha",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--hf-dataset-commit-sha",
        type=str,
        required=False,
    )

    return parser


if __name__ == "__main__":
    args = get_argument_parser().parse_args()

    aiplatform.init(project=args.project_id, location=args.region)

    if args.github_commit_sha is None:
        print("warning: `github_commit_sha` is not provided")

    if args.hf_dataset_commit_sha is None:
        print("warning: `hf_dataset_commit_sha` is not provided")

    pipeline_job = PipelineJob(
        display_name="Ainu LM via Pull Request",
        template_path=str(pipeline_path),
        job_id=f"ainu-lm-pull-request-{get_timestamp()}",
        pipeline_root=args.pipeline_root,
        parameter_values={
            "github_commit_sha": args.github_commit_sha,
            "hf_dataset_commit_sha": args.hf_dataset_commit_sha,
        },
    )

    pipeline_job.submit(
        service_account=args.service_account,
    )
