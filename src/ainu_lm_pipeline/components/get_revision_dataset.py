from typing import Optional

from kfp import dsl


@dsl.component(
    base_image="python:3.10",
    packages_to_install=["huggingface-hub"],
)
def get_revision_dataset(
    hf_repo_id: str,
    hf_token: str,
    hf_dataset_commit_sha: Optional[str] = None,
) -> str:
    from huggingface_hub import HfApi

    if hf_dataset_commit_sha is not None:
        return hf_dataset_commit_sha

    hf_api = HfApi(token=hf_token)
    hf_repo = hf_api.repo_info(hf_repo_id, repo_type="dataset")

    return hf_repo.sha
