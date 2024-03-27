import argparse
import os

from kfp.registry import RegistryClient

from .config import pipeline_path


def get_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", type=str, default=os.getenv("PROJECT_ID"))
    parser.add_argument("--region", type=str, default=os.getenv("REGION"))
    parser.add_argument(
        "--kfp-repo",
        type=str,
        default=os.getenv("KFP_REPO"),
        help="KFP repository name. e.g. my-kfp-repo",
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        default=["latest"],
    )
    return parser


if __name__ == "__main__":
    args = get_argument_parser().parse_args()

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
