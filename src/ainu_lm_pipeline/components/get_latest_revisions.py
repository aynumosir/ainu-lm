from typing import NamedTuple

from kfp import dsl


@dsl.component(
    base_image="python:3.10",
    packages_to_install=[
        "huggingface_hub",
        "pygithub",
    ],
)
def get_latest_revisions(
    github_repo_id: str,
    github_token: str,
    hf_repo_id: str,
    hf_token: str,
) -> NamedTuple(
    "Output",
    [
        ("github_repo_sha", str),
        ("hf_repo_sha", str),
    ],
):
    from github import Github
    from huggingface_hub import HfApi

    github = Github(github_token)
    hf_api = HfApi(token=hf_token)

    github_repo = github.get_repo(github_repo_id)
    github_repo_sha = github_repo.get_commits()[0].sha

    hf_repo = hf_api.repo_info(hf_repo_id, repo_type="dataset")
    hf_repo_sha = hf_repo.sha

    return (github_repo_sha, hf_repo_sha)
