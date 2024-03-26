from kfp import dsl


@dsl.component(
    base_image="python:3.10",
    packages_to_install=["huggingface-hub"],
)
def push_to_huggingface_hub(
    project_id: str,
    model_gcs_path: str,
    hf_repo: str,
    hf_token: str,
) -> None:
    from huggingface_hub import HfFolder, Repository

    model_path = model_gcs_path.replace("gs://", "/gcs/")

    # Initialize Hugging Face repository
    repo = Repository(
        project=project_id,
        name=hf_repo,
        token=hf_token,
    )

    # Push model to Hugging Face Hub
    repo.push_from_folder(HfFolder(folder=model_path))
