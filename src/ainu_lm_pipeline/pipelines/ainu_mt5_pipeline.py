from typing import Optional

from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp
from kfp import dsl

from ..components import common, get_mt5_training_job_spec


@dsl.pipeline(name="ainu-mt5-pipeline", pipeline_root="ainu-lm")
def ainu_mt5_pipeline(
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
    hf_dataset_commit_sha: Optional[str] = None,
    github_commit_sha: Optional[str] = None,
    push_to_hub: Optional[bool] = False,
) -> None:
    # ----------------------------------------------------
    # トークンの取得
    # ----------------------------------------------------
    get_hf_token_op = common.get_secret_by_id(
        project_id=project_id,
        secret_id=hf_secret_id,
    ).set_display_name("Hugging Face Hub のトークン取得")

    get_github_token_op = common.get_secret_by_id(
        project_id=project_id,
        secret_id=github_secret_id,
    ).set_display_name("GitHub のトークン取得")

    # ----------------------------------------------------
    # リビジョンの取得
    # ----------------------------------------------------
    get_source_revision_op = (
        common.get_source_revision(
            github_repo_id=github_repo,
            github_token=get_github_token_op.output,
            github_commit_sha=github_commit_sha,
        )
        .set_display_name("GitHubのリビジョンの取得")
        .set_caching_options(False)
    )

    get_dataset_revision_op = (
        common.get_dataset_revision(
            hf_repo_id=hf_dataset_repo,
            hf_token=get_hf_token_op.output,
            hf_dataset_commit_sha=hf_dataset_commit_sha,
        )
        .set_display_name("Hugging Face Datasetsのリビジョンの取得")
        .set_caching_options(False)
    )

    training_job_suffix = (
        f"{get_source_revision_op.output}-{get_dataset_revision_op.output}"
    )

    # ----------------------------------------------------
    # 出力ディレクトリの取得
    # ----------------------------------------------------
    get_base_output_directory_op = common.get_base_output_directory(
        pipeline_staging=pipeline_staging,
        source_sha=get_source_revision_op.output,
        dataset_sha=get_dataset_revision_op.output,
    ).set_display_name("出力ディレクトリの取得")

    # ----------------------------------------------------
    # カスタム訓練イメージのビルド
    # ----------------------------------------------------
    build_custom_train_image_op = common.build_trainer_image(
        project_id=project_id,
        training_image_uri=train_image_uri,
        github_repo=github_repo,
        github_commit_sha=get_source_revision_op.output,
        hf_token=get_hf_token_op.output,
    ).set_display_name("カスタム訓練イメージのビルド")

    # ----------------------------------------------------
    # mT5 訓練ジョブの仕様を取得
    # ----------------------------------------------------
    get_mt5_training_job_spec_op = (
        get_mt5_training_job_spec(
            train_image_uri=train_image_uri,
            push_to_hub=push_to_hub,
            dataset_revision=get_dataset_revision_op.output,
        )
        .after(build_custom_train_image_op)
        .set_display_name("mT5訓練ジョブの仕様を取得")
    )

    # ----------------------------------------------------
    # mT5の訓練
    # ----------------------------------------------------
    CustomTrainingJobOp(
        project=project_id,
        display_name=f"ainu-lm-mt5-{training_job_suffix}",
        base_output_directory=get_base_output_directory_op.output,
        worker_pool_specs=get_mt5_training_job_spec_op.output,
        location=location,
        tensorboard=f"projects/{project_id}/locations/{location}/tensorboards/{tensorboard_id}",
        service_account=service_account,
    ).set_display_name("mT5の訓練")
