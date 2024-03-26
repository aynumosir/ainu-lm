from kfp import dsl


@dsl.component(
    base_image="python:3.10",
    packages_to_install=["google-cloud-secret-manager"],
    output_component_file="./dist/get_hf_token.yaml",
)
def get_hf_token(project_id: str, hf_secret_id: str) -> str:
    from google.cloud import secretmanager

    secret_client = secretmanager.SecretManagerServiceClient()
    secret_name = f"projects/{project_id}/secrets/{hf_secret_id}/versions/latest"
    response = secret_client.access_secret_version(name=secret_name)
    return response.payload.data.decode("UTF-8")
