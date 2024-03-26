from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp
from kfp import dsl

from ..components import (
    build_trainer_image,
    get_base_output_directory,
    get_latest_secret_by_id,
    get_lm_training_job_result,
    get_lm_training_job_spec,
    get_revision_dataset,
    get_revision_source,
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
    hf_model_repo: str,
    hf_dataset_repo: str,
    hf_secret_id: str,
    github_repo: str,
    github_secret_id: str,
) -> None:
    # ----------------------------------------------------
    # トークンの取得
    # ----------------------------------------------------
    get_hf_token_op = get_latest_secret_by_id(
        project_id=project_id,
        secret_id=hf_secret_id,
    ).set_display_name("Hugging Face Hub のトークン取得")

    get_github_token_op = get_latest_secret_by_id(
        project_id=project_id,
        secret_id=github_secret_id,
    ).set_display_name("GitHub のトークン取得")

    # ----------------------------------------------------
    # リビジョンの取得
    # ----------------------------------------------------
    get_revision_source_op = (
        get_revision_source(
            github_repo_id=github_repo,
            github_token=get_github_token_op.output,
        )
        .set_display_name("GitHubのリビジョンの取得")
        .set_caching_options(False)
    )

    get_revision_dataset_op = (
        get_revision_dataset(
            hf_repo_id=hf_dataset_repo,
            hf_token=get_hf_token_op.output,
        )
        .set_display_name("Hugging Face Datasetsのリビジョンの取得")
        .set_caching_options(False)
    )

    # ----------------------------------------------------
    # 出力ディレクトリの取得
    # ----------------------------------------------------
    get_base_output_directory_op = get_base_output_directory(
        pipeline_staging=pipeline_staging,
        source_sha=get_revision_source_op.output,
        dataset_sha=get_revision_dataset_op.output,
    ).set_display_name("出力ディレクトリの取得")

    # ----------------------------------------------------
    # カスタム訓練イメージのビルド
    # ----------------------------------------------------
    build_custom_train_image_op = build_trainer_image(
        project_id=project_id,
        training_image_uri=train_image_uri,
        github_repo=github_repo,
        github_commit_sha=get_revision_source_op.output,
        hf_token=get_hf_token_op.output,
    ).set_display_name("カスタム訓練イメージのビルド")

    # ----------------------------------------------------
    # トークナイザの訓練ジョブの仕様を取得
    # ----------------------------------------------------
    get_tokenizer_training_job_spec_op = (
        get_tokenizer_training_job_spec(
            train_image_uri=train_image_uri,
            dataset_revision=get_revision_dataset_op.output,
        )
        .set_display_name("トークナイザの訓練ジョブの仕様を取得")
        .after(build_custom_train_image_op)
    )

    # ----------------------------------------------------
    # トークナイザの訓練
    # ----------------------------------------------------
    tokenizer_training_job_op = CustomTrainingJobOp(
        display_name="ainu-lm-tokenizer",
        base_output_directory=get_base_output_directory_op.output,
        worker_pool_specs=get_tokenizer_training_job_spec_op.output,
    ).set_display_name("トークナイザの訓練")

    # ----------------------------------------------------
    # トークナイザの結果取得
    # ----------------------------------------------------
    get_tokenizer_training_job_result_op = get_tokenizer_training_job_result(
        location=location, job_resource=tokenizer_training_job_op.output
    ).set_display_name("トークナイザの結果取得")

    # ----------------------------------------------------
    # LM訓練ジョブの仕様を取得
    # ----------------------------------------------------
    get_lm_training_job_spec_op = get_lm_training_job_spec(
        train_image_uri=train_image_uri,
        tokenizer_gcs_path=get_tokenizer_training_job_result_op.outputs[
            "model_artifacts"
        ],
        dataset_revision=get_revision_dataset_op.output,
    ).set_display_name("LM訓練ジョブの仕様を取得")

    # ----------------------------------------------------
    # LMの訓練
    # ----------------------------------------------------
    lm_training_job_op = CustomTrainingJobOp(
        project=project_id,
        display_name="ainu-lm-lm",
        base_output_directory=get_base_output_directory_op.output,
        worker_pool_specs=get_lm_training_job_spec_op.output,
        location=location,
        tensorboard=tensorboard_name,
        service_account=service_account,
    ).set_display_name("LMの訓練")

    # ----------------------------------------------------
    # LMの結果取得
    # ----------------------------------------------------
    get_lm_training_job_op = get_lm_training_job_result(
        location=location,
        job_resource=lm_training_job_op.output,
    ).set_display_name("LMの結果取得")

    # ----------------------------------------------------
    # Huggingface Hub に push
    # ----------------------------------------------------
    (
        push_to_huggingface_hub(
            model_gcs_path=get_lm_training_job_op.outputs["model_artifacts"],
            commit_message=f"Update model for {get_revision_dataset_op.output}",
            hf_repo=hf_model_repo,
            hf_token=get_hf_token_op.output,
        ).set_display_name("Hugging Face Hub に push")
    )
