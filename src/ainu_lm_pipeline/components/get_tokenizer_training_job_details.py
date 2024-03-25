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
    from collections import namedtuple

    from google.cloud.aiplatform.gapic import JobServiceClient
    from google.protobuf.json_format import Parse
    from google_cloud_pipeline_components.proto.gcp_resources_pb2 import GcpResources

    client_options = {"api_endpoint": f"{location}-aiplatform.googleapis.com"}
    client = JobServiceClient(client_options=client_options)

    training_gcp_resources = Parse(job_resource, GcpResources())
    resource_uri = training_gcp_resources.resources[0].resource_uri
    resource_name = "/".join(resource_uri.split("/")[4:])

    custom_job = client.get_custom_job(name=resource_name)
    model_artifacts = custom_job.job_spec.base_output_directory.output_uri_prefix

    outputs = namedtuple("Outputs", ["model_artifacts"])
    return outputs(model_artifacts=model_artifacts)
