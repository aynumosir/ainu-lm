import argparse
import os

from google.cloud import aiplatform
from google.cloud.aiplatform import PipelineJob
from google.cloud.aiplatform.pipeline_job_schedules import PipelineJobSchedule

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
        type=str,
        default="https://us-central1-kfp.pkg.dev/neetlab/kfp/ainu-lm-pipeline/latest",
    )
    return parser


if __name__ == "__main__":
    args = get_argument_parser().parse_args()
    aiplatform.init(project=args.project_id, location=args.region)

    pipeline_job = PipelineJob(
        display_name="Ainu LM Pipeline",
        template_path=args.template_path,
        job_id=f"ainu-lm-weekly-{get_timestamp()}",
        pipeline_root=args.pipeline_root,
        parameter_values={
            "push_to_hub": True,
        },
    )

    display_name = "Ainu LM Pipeline Weekly Schedule"
    existing_schedules = PipelineJobSchedule.list(
        filter=f'display_name="{display_name}"',
        order_by="create_time desc",
    )

    if len(existing_schedules) > 0:
        for existing_schedule in existing_schedules:
            existing_schedule.delete(sync=True)

    pipeline_job.create_schedule(
        cron="TZ=Asia/Tokyo 0 0 * * SUN",
        display_name=display_name,
        service_account=args.service_account,
    )
