import os
from argparse import ArgumentParser

from google.cloud import aiplatform
from google.cloud.aiplatform import PipelineJob
from google.cloud.aiplatform.pipeline_job_schedules import PipelineJobSchedule

from .utils import get_template_path, get_timestamp

common = ArgumentParser(add_help=False)
common.add_argument("--project-id", type=str, default=os.getenv("PROJECT_ID"))
common.add_argument("--region", type=str, default=os.getenv("REGION"))
common.add_argument("--pipeline-root", type=str, default=os.getenv("PIPELINE_ROOT"))
common.add_argument("--service-account", type=str, default=os.getenv("SERVICE_ACCOUNT"))
common.add_argument("--cron", type=str, required=True)

parser = ArgumentParser()
subparser = parser.add_subparsers(dest="pipeline")
subparser.add_parser("roberta", parents=[common])
subparser.add_parser("gpt2", parents=[common])
subparser.add_parser("mt5", parents=[common])
subparser.add_parser("mt5-gec", parents=[common])

if __name__ == "__main__":
    args = parser.parse_args()

    aiplatform.init(project=args.project_id, location=args.region)

    pipeline_job = PipelineJob(
        display_name=f"Ainu {args.pipeline.capitalize()} Pipeline Job",
        job_id=f"ainu-{args.pipeline}-job-{get_timestamp()}",
        pipeline_root=args.pipeline_root,
        template_path=get_template_path(region=args.region, pipeline=args.pipeline),
        parameter_values={
            "push_to_hub": True,
        },
    )

    display_name = f"Ainu {args.pipeline.capitalize()} Pipeline Schedule"
    existing_schedules = PipelineJobSchedule.list(
        filter=f'display_name="{display_name}"',
        order_by="create_time desc",
    )

    if len(existing_schedules) > 0:
        for existing_schedule in existing_schedules:
            existing_schedule.delete(sync=True)

    pipeline_job.create_schedule(
        cron=args.cron,
        display_name=display_name,
        service_account=args.service_account,
    )
