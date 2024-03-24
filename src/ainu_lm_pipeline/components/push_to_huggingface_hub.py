from kfp import dsl


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "google-cloud-pipeline-components",
        "google-cloud-aiplatform",
        "huggingface-hub",
        "torch",
        "transformers",
    ],
    output_component_file="./pipelines/push_to_huggingface_hub.yaml",
)
def push_to_huggingface_hub(
    model_gcs_path: str,
    hf_repo: str,
    hf_token: str,
) -> None:
    import os
    from pathlib import Path

    from google.cloud import storage
    from transformers import RobertaForMaskedLM, RobertaTokenizerFast

    download_path = Path("/tmp/model")
    os.makedirs(download_path, exist_ok=True)

    storage_client = storage.Client()

    bucket = model_gcs_path.split("/")[2]
    prefix = "/".join(model_gcs_path.split("/")[3:])

    blobs = storage_client.list_blobs(bucket, prefix=prefix)
    for blob in blobs:
        target_path = download_path / blob.name.split("/")[-1]
        blob.download_to_filename(target_path)

    model = RobertaForMaskedLM.from_pretrained(download_path, local_files_only=True)
    tokenizer = RobertaTokenizerFast.from_pretrained(
        download_path, local_files_only=True
    )

    model.push_to_hub(hf_repo, token=hf_token)
    tokenizer.push_to_hub(hf_repo, token=hf_token)
