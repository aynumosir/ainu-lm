from kfp import dsl


@dsl.component(
    base_image="python:3.10",
    packages_to_install=["huggingface-hub"],
)
def push_to_huggingface_hub(
    model_gcs_path: str,
    commit_message: str,
    hf_repo: str,
    hf_token: str,
) -> None:
    from huggingface_hub import HfApi

    api = HfApi()

    model_path = model_gcs_path.replace("gs://", "/gcs/")

    api.upload_folder(
        repo_id=hf_repo,
        repo_type="model",
        folder_path=model_path,
        commit_message=commit_message,
        token=hf_token,
    )
