from kfp import dsl


@dsl.component(
    base_image="python:3.10",
    packages_to_install=["google-cloud-secret-manager"],
)
def get_secret_by_id(project_id: str, secret_id: str, version: str = "latest") -> str:
    from google.cloud import secretmanager

    secret_client = secretmanager.SecretManagerServiceClient()
    secret_name = f"projects/{project_id}/secrets/{secret_id}/versions/{version}"
    response = secret_client.access_secret_version(name=secret_name)
    return response.payload.data.decode("UTF-8")
