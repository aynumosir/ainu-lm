from google.cloud.aiplatform import hyperparameter_tuning as hpt
from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp
from google_cloud_pipeline_components.v1.hyperparameter_tuning_job import (
    HyperparameterTuningJobRunOp,
    serialize_metrics,
    serialize_parameters,
)
from kfp import dsl

from . import config as cfg
from .components import (
    build_trainer_image,
    get_lm_training_job_details,
    get_tokenizer_training_job_details,
    get_worker_pool_specs,
    push_to_huggingface_hub,
)


@dsl.pipeline(name="ainu-lm-pipeline", pipeline_root="ainu-lm")
def ainu_lm_pipeline(
    pipeline_job_id: str,
    pipeline_staging: str,
    source_repo_name: str,
    source_commit_sha: str,
    tensorboard_id: str,
    hf_repo: str,
    hf_token: str,
) -> None:
    BASE_OUTPUT_DIR = f"{pipeline_staging}/{source_commit_sha}"

    # ----------------------------------------------------
    # カスタム訓練イメージのビルド
    # ----------------------------------------------------
    build_custom_train_image_task = (
        build_trainer_image(
            project_id=cfg.PROJECT_ID,
            training_image_uri=cfg.TRAIN_IMAGE_URI,
            repo_name=source_repo_name,
            commit_sha=source_commit_sha,
            hf_token=hf_token,
        )
        .set_display_name("カスタム訓練イメージのビルド")
        .set_caching_options(True)
    )

    # ----------------------------------------------------
    # トークナイザの訓練
    # ----------------------------------------------------
    tokenizer_training_task = (
        CustomTrainingJobOp(
            display_name=f"{pipeline_job_id}-tokenizer",
            base_output_directory=BASE_OUTPUT_DIR,
            worker_pool_specs=[
                {
                    "container_spec": {
                        "image_uri": cfg.TRAIN_IMAGE_URI,
                        "args": ["tokenizer"],
                    },
                    "machine_spec": {
                        "machine_type": "n1-standard-4",
                    },
                    "replica_count": "1",
                    "disk_spec": {
                        "boot_disk_type": "pd-ssd",
                        "boot_disk_size_gb": 100,
                    },
                }
            ],
        )
        .set_display_name("トークナイザの訓練")
        .set_caching_options(True)
        .after(build_custom_train_image_task)
    )

    # ----------------------------------------------------
    # トークナイザ訓練ジョブの詳細情報取得
    # ----------------------------------------------------
    tokenizer_training_job_details_task = (
        get_tokenizer_training_job_details(
            location=cfg.REGION, job_resource=tokenizer_training_task.output
        )
        .set_display_name("トークナイザ訓練ジョブの詳細情報取得")
        .set_caching_options(True)
    )

    # ----------------------------------------------------
    # worker_pool_specsの定義
    # ----------------------------------------------------
    worker_pool_specs_task = (
        get_worker_pool_specs(
            train_image_uri=cfg.TRAIN_IMAGE_URI,
            tensorboard_id=tensorboard_id,
            tensorboard_experiment_name=pipeline_job_id,
            tokenizer_gcs_path=tokenizer_training_job_details_task.outputs[
                "model_artifacts"
            ],
        )
        .set_display_name("worker_pool_specsの定義")
        .set_caching_options(True)
    )

    # ----------------------------------------------------
    # モデルの訓練
    # ----------------------------------------------------
    lm_training_task = (
        HyperparameterTuningJobRunOp(
            display_name=f"{pipeline_job_id}-lm",
            base_output_directory=BASE_OUTPUT_DIR,
            worker_pool_specs=worker_pool_specs_task.output,
            study_spec_metrics=serialize_metrics(
                {
                    "loss": "minimize",
                }
            ),
            study_spec_parameters=serialize_parameters(
                {
                    "num-train-epochs": hpt.IntegerParameterSpec(
                        min=10, max=20, scale="linear"
                    ),
                }
            ),
            max_trial_count=1,
            parallel_trial_count=1,
            location=cfg.REGION,
            project=cfg.PROJECT_ID,
        )
        .set_display_name("モデルの訓練")
        .set_caching_options(True)
        .after(tokenizer_training_task)
    )

    # ----------------------------------------------------
    # 訓練ジョブの詳細情報取得
    # ----------------------------------------------------
    training_job_details_task = get_lm_training_job_details(
        location=cfg.REGION,
        job_resource=lm_training_task.output,
    ).set_display_name("訓練ジョブの詳細情報取得")

    # ----------------------------------------------------
    # Huggingface Hub に push
    # ----------------------------------------------------
    (
        push_to_huggingface_hub(
            model_gcs_path=training_job_details_task.outputs["model_artifacts"],
            hf_repo=hf_repo,
            hf_token=hf_token,
        ).set_display_name("Huggingface Hub に push")
    )
