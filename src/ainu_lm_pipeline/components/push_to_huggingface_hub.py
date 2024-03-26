from kfp import dsl


@dsl.component(
    base_image="python:3.10",
    packages_to_install=[
        "google-cloud-pipeline-components",
        "huggingface-hub",
        "torch",
        "transformers",
    ],
)
def push_to_huggingface_hub(
    project_id: str,
    model_gcs_path: str,
    hf_repo: str,
    hf_token: str,
) -> None:
    from transformers import RobertaForMaskedLM, RobertaTokenizerFast

    model_path = model_gcs_path.replace("gs://", "/gcs/")

    model = RobertaForMaskedLM.from_pretrained(model_path, local_files_only=True)
    tokenizer = RobertaTokenizerFast.from_pretrained(model_path, local_files_only=True)

    model.push_to_hub(hf_repo, token=hf_token)
    tokenizer.push_to_hub(hf_repo, token=hf_token)
