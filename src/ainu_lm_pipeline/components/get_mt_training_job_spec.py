from kfp import dsl


@dsl.component(base_image="python:3.10")
def get_mt_training_job_spec(
    train_image_uri: str,
    dataset_name: str,
    dataset_revision: str,
    hub_model_id: str,
    push_to_hub: bool,
) -> list:
    worker_pool_specs = [
        {
            "container_spec": {
                "image_uri": train_image_uri,
                "args": [
                    "train",
                    "mt",
                    "--base-model=google/mt5-small",
                    f"--dataset-name={dataset_name}",
                    f"--dataset-revision={dataset_revision}",
                    "--num-train-epochs=20",
                    "--per-device-train-batch-size=16",
                    "--per-device-eval-batch-size=16",
                    "--gradient-accumulation-steps=2",
                    "--learning-rate=5e-4",
                    "--warmup-ratio=0.06",
                    "--weight-decay=0.01",
                    "--experiment-task-prefix=all",
                    f"--hub-model-id={hub_model_id}",
                    f"--push-to-hub={'yes' if push_to_hub else 'no'}",
                ],
            },
            # https://cloud.google.com/vertex-ai/docs/training/configure-compute?hl=ja#specifying_gpus
            "machine_spec": {
                "machine_type": "a2-highgpu-1g",
                "accelerator_type": "NVIDIA_TESLA_A100",
                "accelerator_count": 1,
            },
            "replica_count": "1",
            "disk_spec": {
                "boot_disk_type": "pd-ssd",
                "boot_disk_size_gb": 100,
            },
        }
    ]

    return worker_pool_specs
