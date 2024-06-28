import os
from argparse import ArgumentParser

from kfp.compiler import Compiler

from ..pipelines import (
    ainu_gpt2_pipeline,
    ainu_mt5_affix_pipeline,
    ainu_mt5_gec_pipeline,
    ainu_mt5_pipeline,
    ainu_roberta_pipeline,
)
from .utils import get_pipeline_path

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
# fmt: on

parser = ArgumentParser()
subparsers = parser.add_subparsers(dest="pipeline")
subparsers.add_parser("roberta", parents=[common])
subparsers.add_parser("gpt2", parents=[common])
subparsers.add_parser("mt5", parents=[common])
subparsers.add_parser("mt5-gec", parents=[common])
subparsers.add_parser("mt5-affix", parents=[common])


if __name__ == "__main__":
    args = parser.parse_args()

    pipeline_path = get_pipeline_path(args.pipeline)
    os.makedirs(pipeline_path.parent, exist_ok=True)

    if args.pipeline == "roberta":
        pipeline_func = ainu_roberta_pipeline
    elif args.pipeline == "gpt2":
        pipeline_func = ainu_gpt2_pipeline
    elif args.pipeline == "mt5":
        pipeline_func = ainu_mt5_pipeline
    elif args.pipeline == "mt5-gec":
        pipeline_func = ainu_mt5_gec_pipeline
    elif args.pipeline == "mt5-affix":
        pipeline_func = ainu_mt5_affix_pipeline

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
