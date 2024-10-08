import os
from argparse import ArgumentParser, Namespace

from kfp.compiler import Compiler

from ...pipelines import ainu_kana_pipeline, ainu_mt_pipeline
from ..utils import get_pipeline_path


def add_parser(parser: ArgumentParser) -> None:
    # fmt: off
    common = ArgumentParser(add_help=False)
    common.add_argument("--project-id", type=str, default=os.getenv("PROJECT_ID"))
    common.add_argument("--region", type=str, default=os.getenv("REGION"))
    common.add_argument("--train-image-uri", type=str, default=os.getenv("TRAIN_IMAGE_URI"))
    common.add_argument("--pipeline-staging", type=str, default=os.getenv("PIPELINE_STAGING"))
    common.add_argument("--tensorboard-id", type=str, default=os.getenv("TENSORBOARD_ID"))
    common.add_argument("--service-account", type=str, default=os.getenv("SERVICE_ACCOUNT"))
    common.add_argument("--hf-dataset-repo", type=str, default=os.getenv("HF_DATASET_REPO"))
    common.add_argument("--hf-secret-id", type=str, default=os.getenv("HF_SECRET_ID"))
    common.add_argument("--github-repo", type=str, default=os.getenv("GITHUB_REPO"))
    common.add_argument("--github-secret-id", type=str, default=os.getenv("GITHUB_SECRET_ID"))

    subparsers = parser.add_subparsers(dest="pipeline")
    subparsers.add_parser("mt", parents=[common])
    subparsers.add_parser("kana", parents=[common])
    # fmt: on


def main(args: Namespace) -> None:
    pipeline_path = get_pipeline_path(args.pipeline)

    os.makedirs(pipeline_path.parent, exist_ok=True)

    if args.pipeline == "mt":
        pipeline_func = ainu_mt_pipeline
    elif args.pipeline == "kana":
        pipeline_func = ainu_kana_pipeline

    compiler = Compiler()
    compiler.compile(
        pipeline_func=pipeline_func,
        package_path=str(pipeline_path),
        pipeline_parameters={
            "project_id": args.project_id,
            "location": args.region,
            "train_image_uri": args.train_image_uri,
            "pipeline_staging": args.pipeline_staging,
            "tensorboard_id": args.tensorboard_id,
            "service_account": args.service_account,
            "hf_dataset_repo": args.hf_dataset_repo,
            "hf_secret_id": args.hf_secret_id,
            "github_repo": args.github_repo,
            "github_secret_id": args.github_secret_id,
        },
    )
