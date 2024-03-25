from typing import NamedTuple

from kfp import dsl


@dsl.component(
    base_image="python:3.10",
    packages_to_install=[
        "google-cloud-pipeline-components",
        "google-cloud-aiplatform",
        "pandas",
    ],
    output_component_file="./pipelines/get_lm_training_job_details.yaml",
)
def get_lm_training_job_details(
    location: str,
    job_resource: str,
) -> NamedTuple(
    "Outputs",
    [
        ("model_artifacts", str),
        ("eval_loss", float),
    ],
):
    from collections import namedtuple
    from typing import Sequence

    import pandas as pd
    from google.cloud.aiplatform.gapic import JobServiceClient
    from google.cloud.aiplatform_v1.types import study
    from google.protobuf.json_format import Parse
    from google_cloud_pipeline_components.proto.gcp_resources_pb2 import GcpResources

    class HyperparameterTrainingJobTrials(pd.DataFrame):
        def __init__(self, trials: Sequence[study.Trial]) -> None:
            results = []

            for trial in trials:
                metric = next(
                    (
                        metric.value
                        for metric in trial.final_measurement.metrics
                        if metric.metric_id == "loss"
                    ),
                    None,
                )

                results.append(
                    {
                        "id": trial.id,
                        "name": trial.name,
                        "state": trial.state,
                        "parameters": trial.parameters,
                        "final_measurement": metric,
                        "measurements": trial.measurements,
                    }
                )

            super().__init__(results)

        @property
        def best(self) -> study.Trial:
            return self.loc[self["final_measurement"].idxmin()]

    client_options = {"api_endpoint": f"{location}-aiplatform.googleapis.com"}
    client = JobServiceClient(client_options=client_options)

    training_gcp_resources = Parse(job_resource, GcpResources())
    resource_uri = training_gcp_resources.resources[0].resource_uri

    hyperparameter_tuning_job_name = "/".join(resource_uri.split("/")[4:])
    hyperparameter_tuning_job = client.get_hyperparameter_tuning_job(
        name=hyperparameter_tuning_job_name
    )

    trials = HyperparameterTrainingJobTrials(hyperparameter_tuning_job.trials)
    best_trial = trials.best

    prefix = (
        hyperparameter_tuning_job.trial_job_spec.base_output_directory.output_uri_prefix
    )
    model_artifacts_uri = "/".join([prefix, best_trial.id])

    outputs = namedtuple("Outputs", ["model_artifacts", "eval_loss"])
    return outputs(
        model_artifacts=model_artifacts_uri, eval_loss=best_trial.final_measurement
    )
