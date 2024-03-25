from typing import NamedTuple

from kfp import dsl


@dsl.component(
    base_image="gcr.io/google.com/cloudsdktool/cloud-sdk:latest",
    packages_to_install=["google-cloud-build"],
    output_component_file="./dist/build_trainer_image.yaml",
)
def build_trainer_image(
    project_id: str,
    training_image_uri: str,
    repo_name: str,
    commit_sha: str,
    hf_token: str,
) -> NamedTuple("Outputs", [("training_image_uri", str)]):
    from google.cloud.devtools import cloudbuild_v1 as cloudbuild

    build_client = cloudbuild.CloudBuildClient()

    build = cloudbuild.Build(
        source=cloudbuild.Source(
            repo_source=cloudbuild.RepoSource(
                project_id=project_id,
                repo_name=repo_name,
                commit_sha=commit_sha,
            )
        ),
        steps=[
            cloudbuild.BuildStep(
                name="gcr.io/kaniko-project/executor:latest",
                args=[
                    "--context=dir://.",
                    "--destination=$_IMAGE_URI",
                    "--cache=true",
                    "--dockerfile=./src/ainu_lm_trainer/Dockerfile",
                    f"--build-arg=HF_TOKEN={hf_token}",
                ],
            )
        ],
        substitutions={"_IMAGE_URI": training_image_uri},
    )

    operation = build_client.create_build(project_id=project_id, build=build)
    operation.result()

    return (training_image_uri,)
