from kfp import dsl


@dsl.component(base_image="python:3.10")
def get_base_output_directory(
    pipeline_staging: str,
    source_sha: str,
    dataset_sha: str,
) -> str:
    return f"{pipeline_staging}/{source_sha}+{dataset_sha}"
