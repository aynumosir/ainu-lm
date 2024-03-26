from kfp import dsl


@dsl.component(
    base_image="python:3.10",
    packages_to_install=["pygithub"],
)
def get_revision_source(
    github_repo_id: str,
    github_token: str,
) -> str:
    from github import Github

    github = Github(github_token)
    github_repo = github.get_repo(github_repo_id)
    github_repo_sha = github_repo.get_commits()[0].sha

    return github_repo_sha
