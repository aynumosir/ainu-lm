from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp
from kfp import dsl

from ..components import (
    build_trainer_image,
    get_base_output_directory,
    get_latest_revisions,
    get_latest_secret_by_id,
    get_lm_training_job_result,
    get_lm_training_job_spec,
    get_tokenizer_training_job_result,
    get_tokenizer_training_job_spec,
    push_to_huggingface_hub,
)


@dsl.pipeline(name="ainu-lm-pipeline", pipeline_root="ainu-lm")
def ainu_lm_pipeline(
    project_id: str,
    location: str,
    service_account: str,
    tensorboard_name: str,
    train_image_uri: str,
    pipeline_staging: str,
    source_repo_name: str,
    hf_model_repo: str,
    hf_dataset_repo: str,
    hf_secret_id: str,
    github_repo: str,
    github_secret_id: str,
) -> None:
    # ----------------------------------------------------
    # Hugging Face Hub のトークン取得
    # ----------------------------------------------------
    get_hf_token_op = (
        get_latest_secret_by_id(
            project_id=project_id,
            secret_id=hf_secret_id,
        )
        .set_display_name("Hugging Face Hub のトークン取得")
        .set_caching_options(True)
    )

    # ----------------------------------------------------
    # GitHub のトークン取得
    # ----------------------------------------------------
    get_github_token_op = (
        get_latest_secret_by_id(
            project_id=project_id,
            secret_id=github_secret_id,
        )
        .set_display_name("GitHub のトークン取得")
        .set_caching_options(True)
    )

    # ----------------------------------------------------
    # リビジョンの取得
    # ----------------------------------------------------
    get_latest_revisions_op = (
        get_latest_revisions(
            github_repo_id=github_repo,
            github_token=get_github_token_op.output,
            hf_repo_id=hf_dataset_repo,
            hf_token=get_hf_token_op.output,
        )
        .set_display_name("リビジョンの取得")
        .set_caching_options(False)
    )

    # ----------------------------------------------------
    # 出力ディレクトリの取得
    # ----------------------------------------------------
    get_base_output_directory_op = (
        get_base_output_directory(
            pipeline_staging=pipeline_staging,
            source_sha=get_latest_revisions_op.outputs["github_repo_sha"],
            dataset_sha=get_latest_revisions_op.outputs["hf_repo_sha"],
        )
        .set_display_name("出力ディレクトリの取得")
        .set_caching_options(True)
    )

    # ----------------------------------------------------
    # カスタム訓練イメージのビルド
    # ----------------------------------------------------
    build_custom_train_image_op = (
        build_trainer_image(
            project_id=project_id,
            training_image_uri=train_image_uri,
            repo_name=source_repo_name,
            commit_sha=get_latest_revisions_op.outputs["github_repo_sha"],
            hf_token=get_hf_token_op.output,
        )
        .set_display_name("カスタム訓練イメージのビルド")
        .set_caching_options(True)
    )

    # ----------------------------------------------------
    # トークナイザの訓練ジョブの仕様を取得
    # ----------------------------------------------------
    get_tokenizer_training_job_spec_op = (
        get_tokenizer_training_job_spec(
            train_image_uri=train_image_uri,
            dataset_revision=get_latest_revisions_op.outputs["hf_repo_sha"],
        )
        .set_display_name("トークナイザの訓練ジョブの仕様を取得")
        .set_caching_options(True)
        .after(build_custom_train_image_op)
    )

    # ----------------------------------------------------
    # トークナイザの訓練
    # ----------------------------------------------------
    tokenizer_training_job_op = (
        CustomTrainingJobOp(
            display_name="ainu-lm-tokenizer",
            base_output_directory=get_base_output_directory_op.output,
            worker_pool_specs=get_tokenizer_training_job_spec_op.output,
        )
        .set_display_name("トークナイザの訓練")
        .set_caching_options(True)
    )

    # ----------------------------------------------------
    # トークナイザ訓練ジョブの結果取得
    # ----------------------------------------------------
    get_tokenizer_training_job_result_op = (
        get_tokenizer_training_job_result(
            location=location, job_resource=tokenizer_training_job_op.output
        )
        .set_display_name("トークナイザ訓練ジョブの結果取得")
        .set_caching_options(True)
    )

    # ----------------------------------------------------
    # モデル訓練ジョブの仕様を取得
    # ----------------------------------------------------
    get_lm_training_job_spec_op = (
        get_lm_training_job_spec(
            train_image_uri=train_image_uri,
            tokenizer_gcs_path=get_tokenizer_training_job_result_op.outputs[
                "model_artifacts"
            ],
            dataset_revision=get_latest_revisions_op.outputs["hf_repo_sha"],
        )
        .set_display_name("モデル訓練ジョブの仕様を取得")
        .set_caching_options(True)
    )

    # ----------------------------------------------------
    # モデルの訓練
    # ----------------------------------------------------
    lm_training_job_op = (
        CustomTrainingJobOp(
            project=project_id,
            display_name="ainu-lm-lm",
            base_output_directory=get_base_output_directory_op.output,
            worker_pool_specs=get_lm_training_job_spec_op.output,
            location=location,
            tensorboard=tensorboard_name,
            service_account=service_account,
        )
        .set_display_name("モデルの訓練")
        .set_caching_options(True)
    )

    # ----------------------------------------------------
    # 訓練ジョブの詳細情報取得
    # ----------------------------------------------------
    get_lm_training_job_op = (
        get_lm_training_job_result(
            location=location,
            job_resource=lm_training_job_op.output,
        )
        .set_display_name("訓練ジョブの詳細情報取得")
        .set_caching_options(True)
    )

    # ----------------------------------------------------
    # Huggingface Hub に push
    # ----------------------------------------------------
    (
        push_to_huggingface_hub(
            project_id=project_id,
            model_gcs_path=get_lm_training_job_op.outputs["model_artifacts"],
            hf_repo=hf_model_repo,
            hf_token=get_hf_token_op.output,
        )
        .set_display_name("Huggingface Hub に push")
        .set_caching_options(True)
    )
