from kfp import dsl


@dsl.component(
    base_image="python:3.10",
    output_component_file="./dist/get_base_output_directory.yaml",
)
def get_base_output_directory(
    pipeline_staging: str,
    source_sha: str,
    dataset_sha: str,
) -> str:
    return f"{pipeline_staging}/{source_sha}+{dataset_sha}"
