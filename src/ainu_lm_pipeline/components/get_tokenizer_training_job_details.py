from typing import NamedTuple

from kfp import dsl


@dsl.component(
    base_image="python:3.10",
    packages_to_install=[
        "google-cloud-pipeline-components",
        "google-cloud-aiplatform",
    ],
    output_component_file="./dist/get_tokenizer_training_job_details.yaml",
)
def get_tokenizer_training_job_details(
    location: str,
    job_resource: str,
) -> NamedTuple(
    "Outputs",
    [
        ("model_artifacts", str),
    ],
):
    from google.cloud.aiplatform.gapic import JobServiceClient
    from google.protobuf.json_format import Parse
    from google_cloud_pipeline_components.proto.gcp_resources_pb2 import GcpResources

    job_client_options = {"api_endpoint": f"{location}-aiplatform.googleapis.com"}
    job_client = JobServiceClient(client_options=job_client_options)

    training_gcp_resources = Parse(job_resource, GcpResources())
    custom_job_id = training_gcp_resources.resources[0].resource_uri
    custom_job_name = "/".join(custom_job_id.split("/")[4:])
    job_resource = job_client.get_custom_job(name=custom_job_name)
    job_base_dir = job_resource.job_spec.base_output_directory.output_uri_prefix

    return (job_base_dir,)
