import os
from argparse import ArgumentParser, Namespace

from kfp.registry import RegistryClient

from ..utils import get_pipeline_path


def add_parser(parser: ArgumentParser) -> None:
    # fmt: off
    common = ArgumentParser(add_help=False)
    common.add_argument("--project-id", type=str, default=os.getenv("PROJECT_ID"))
    common.add_argument("--region", type=str, default=os.getenv("REGION"))
    common.add_argument("--kfp-repo", type=str, default=os.getenv("KFP_REPO"))
    common.add_argument("--tags", type=str, nargs="+", default=["latest"])

    subparsers = parser.add_subparsers(dest="pipeline")
    subparsers.add_parser("mt", parents=[common])
    subparsers.add_parser("kana", parents=[common])
    # fmt: on


def main(args: Namespace) -> None:
    pipeline_path = get_pipeline_path(args.pipeline)

    registry_client = RegistryClient(
        host=f"https://{args.region}-kfp.pkg.dev/{args.project_id}/{args.kfp_repo}"
    )

    template_name, version_name = registry_client.upload_pipeline(
        file_name=str(pipeline_path),
        tags=args.tags,
    )

    print("~" * 80)
    print(f"Pipeline template: {template_name}")
    print(f"Pipeline version: {version_name}")
    print("~" * 80)
