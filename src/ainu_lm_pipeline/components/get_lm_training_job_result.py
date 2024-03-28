from typing import NamedTuple

from kfp import dsl


@dsl.component(
    base_image="python:3.10",
    packages_to_install=[
        "google-cloud-pipeline-components",
        "google-cloud-aiplatform",
        "pandas",
        "pyarrow",
    ],
)
def get_lm_training_job_result(
    location: str,
    job_resource: str,
    model: dsl.Output[dsl.Model],
) -> NamedTuple(
    "Outputs",
    [
        ("model_artifacts", str),
    ],
):
    import shutil

    import pandas as pd
    from google.cloud.aiplatform.gapic import JobServiceClient
    from google.protobuf.json_format import Parse
    from google_cloud_pipeline_components.proto.gcp_resources_pb2 import GcpResources

    # Initialize client
    job_client_options = {"api_endpoint": f"{location}-aiplatform.googleapis.com"}
    job_client = JobServiceClient(client_options=job_client_options)

    # Get custom job
    training_gcp_resources = Parse(job_resource, GcpResources())
    custom_job_id = training_gcp_resources.resources[0].resource_uri
    custom_job_name = "/".join(custom_job_id.split("/")[4:])
    job_resource = job_client.get_custom_job(name=custom_job_name)
    job_base_dir = job_resource.job_spec.base_output_directory.output_uri_prefix

    model_path = f"{job_base_dir}/model"
    checkpoints_path = f"{job_base_dir}/checkpoints"

    # Copy model artifacts
    shutil.copytree(checkpoints_path.replace("gs://", "/gcs/"), model.path)

    # Fetch metrics
    metrics_uri = f"{model.path}/all_results.json"
    metrics_df = pd.read_json(metrics_uri, typ="series")

    # Set model metadata
    model.metadata = {
        "framework": "pytorch",
        "job_name": custom_job_name,
        "epoch": metrics_df["epoch"],
        "time_to_train_in_seconds": (
            job_resource.end_time - job_resource.start_time
        ).total_seconds(),
    }

    return (model_path,)
