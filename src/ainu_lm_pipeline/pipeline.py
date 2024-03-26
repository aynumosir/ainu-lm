from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp
from kfp import dsl

from .components import (
    build_trainer_image,
    get_hf_token,
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
    pipeline_job_id: str,
    pipeline_staging: str,
    source_repo_name: str,
    source_commit_sha: str,
    hf_repo: str,
    hf_secret_id: str,
) -> None:
    BASE_OUTPUT_DIR = f"{pipeline_staging}/{source_commit_sha}"

    # ----------------------------------------------------
    # Hugging Face Hub のトークン取得
    # ----------------------------------------------------
    get_hf_token_op = (
        get_hf_token(
            project_id=project_id,
            hf_secret_id=hf_secret_id,
        )
        .set_display_name("Hugging Face Hub のトークン取得")
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
            commit_sha=source_commit_sha,
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
        )
        .set_display_name("トークナイザの訓練ジョブの仕様を取得")
        .set_caching_options(True)
    )

    # ----------------------------------------------------
    # トークナイザの訓練
    # ----------------------------------------------------
    tokenizer_training_job_op = (
        CustomTrainingJobOp(
            display_name=f"{pipeline_job_id}-tokenizer",
            base_output_directory=BASE_OUTPUT_DIR,
            worker_pool_specs=get_tokenizer_training_job_spec_op.output,
        )
        .set_display_name("トークナイザの訓練")
        .set_caching_options(True)
        .after(build_custom_train_image_op)
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
            display_name=f"{pipeline_job_id}-lm",
            base_output_directory=BASE_OUTPUT_DIR,
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
    get_lm_training_job_op = get_lm_training_job_result(
        location=location,
        job_resource=lm_training_job_op.output,
    ).set_display_name("訓練ジョブの詳細情報取得")

    # ----------------------------------------------------
    # Huggingface Hub に push
    # ----------------------------------------------------
    (
        push_to_huggingface_hub(
            project_id=project_id,
            model_gcs_path=get_lm_training_job_op.outputs["model_artifacts"],
            hf_repo=hf_repo,
            hf_token=get_hf_token_op.output,
        ).set_display_name("Huggingface Hub に push")
    )
