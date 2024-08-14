from typing import Optional

from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp
from kfp import dsl

from ..components import (
    common,
    get_mt_training_job_spec,
)


@dsl.pipeline(name="ainu_mt_pipeline", pipeline_root="ainu-lm")
def ainu_mt_pipeline(
    project_id: str,
    location: str,
    service_account: str,
    tensorboard_id: str,
    train_image_uri: str,
    pipeline_staging: str,
    hf_dataset_repo: str,
    hf_secret_id: str,
    github_repo: str,
    github_secret_id: str,
    push_to_hub: bool = False,
    hf_dataset_commit_sha: Optional[str] = None,
    github_commit_sha: Optional[str] = None,
) -> None:
    get_hf_token_op = (
        common.get_secret_by_id(
            project_id=project_id,
            secret_id=hf_secret_id,
        )
        .set_display_name("HF Hubのトークン取得")
        .set_caching_options(False)
    )

    get_github_token_op = (
        common.get_secret_by_id(
            project_id=project_id,
            secret_id=github_secret_id,
        )
        .set_display_name("GitHubのトークン取得")
        .set_caching_options(False)
    )

    get_source_revision_op = (
        common.get_source_revision(
            github_repo_id=github_repo,
            github_token=get_github_token_op.output,
            github_commit_sha=github_commit_sha,
        )
        .set_display_name("ソースコードのリビジョンの取得")
        .set_caching_options(False)
    )

    get_dataset_revision_op = (
        common.get_dataset_revision(
            hf_repo_id=hf_dataset_repo,
            hf_token=get_hf_token_op.output,
            hf_dataset_commit_sha=hf_dataset_commit_sha,
        )
        .set_display_name("データセットのリビジョンの取得")
        .set_caching_options(False)
    )

    training_job_suffix = (
        f"{get_source_revision_op.output}-{get_dataset_revision_op.output}"
    )

    get_base_output_directory_op = common.get_base_output_directory(
        pipeline_staging=pipeline_staging,
        source_sha=get_source_revision_op.output,
        dataset_sha=get_dataset_revision_op.output,
    ).set_display_name("出力ディレクトリの取得")

    build_custom_train_image_op = common.build_trainer_image(
        project_id=project_id,
        training_image_uri=train_image_uri,
        github_repo=github_repo,
        github_commit_sha=get_source_revision_op.output,
        hf_token=get_hf_token_op.output,
    ).set_display_name("カスタム訓練イメージのビルド")

    get_mt_training_job_spec_op = get_mt_training_job_spec(
        train_image_uri=train_image_uri,
        dataset_name=hf_dataset_repo,
        dataset_revision=get_dataset_revision_op.output,
        hub_model_id="aynumosir/mt5-base-ainu",
        push_to_hub=True,
    ).set_display_name("MTジョブの仕様を取得")

    (
        CustomTrainingJobOp(
            project=project_id,
            display_name=f"ainu-lm-mt-{training_job_suffix}",
            base_output_directory=f"{get_base_output_directory_op.output}",
            worker_pool_specs=get_mt_training_job_spec_op.output,
            location=location,
            tensorboard=f"projects/{project_id}/locations/{location}/tensorboards/{tensorboard_id}",
            service_account=service_account,
        )
        .set_display_name("MTの訓練")
        .after(build_custom_train_image_op)
    )
