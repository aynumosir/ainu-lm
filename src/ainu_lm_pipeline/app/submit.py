import os
from argparse import ArgumentParser

from google.cloud import aiplatform
from google.cloud.aiplatform import PipelineJob

from .utils import get_pipeline_path, get_timestamp

common = ArgumentParser(add_help=False)
common.add_argument("--project-id", type=str, default=os.getenv("PROJECT_ID"))
common.add_argument("--region", type=str, default=os.getenv("REGION"))
common.add_argument("--pipeline-root", type=str, default=os.getenv("PIPELINE_ROOT"))
common.add_argument("--service-account", type=str, default=os.getenv("SERVICE_ACCOUNT"))
common.add_argument("--github-commit-sha", type=str)
common.add_argument("--hf-dataset-commit-sha", type=str)
common.add_argument("--push-to-hub", type=bool, default=False)

parser = ArgumentParser()
subparsers = parser.add_subparsers(dest="pipeline")
subparsers.add_parser("roberta", parents=[common])
subparsers.add_parser("gpt2", parents=[common])
subparsers.add_parser("mt5", parents=[common])
subparsers.add_parser("mt5-gec", parents=[common])


if __name__ == "__main__":
    args = parser.parse_args()

    aiplatform.init(project=args.project_id, location=args.region)

    if args.github_commit_sha is None:
        print("warning: `github_commit_sha` is not provided")

    if args.hf_dataset_commit_sha is None:
        print("warning: `hf_dataset_commit_sha` is not provided")

    pipeline_job = PipelineJob(
        display_name=f"Ainu {args.pipeline.capitalize()} Pull Request Pipeline Job",
        template_path=str(get_pipeline_path(args.pipeline)),
        job_id=f"ainu-{args.pipeline}-pr-{get_timestamp()}",
        pipeline_root=args.pipeline_root,
        parameter_values={
            "github_commit_sha": args.github_commit_sha,
            "hf_dataset_commit_sha": args.hf_dataset_commit_sha,
            "push_to_hub": args.push_to_hub,
        },
    )

    pipeline_job.submit(
        service_account=args.service_account,
    )
