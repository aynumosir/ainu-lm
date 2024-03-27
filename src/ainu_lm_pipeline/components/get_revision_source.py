from typing import Optional

from kfp import dsl


@dsl.component(
    base_image="python:3.10",
    packages_to_install=["pygithub"],
)
def get_revision_source(
    github_repo_id: str,
    github_token: str,
    github_commit_sha: Optional[str] = None,
) -> str:
    from github import Github

    if github_commit_sha is not None:
        return github_commit_sha

    github = Github(github_token)
    github_repo = github.get_repo(github_repo_id)
    github_repo_sha = github_repo.get_commits()[0].sha

    return github_repo_sha
