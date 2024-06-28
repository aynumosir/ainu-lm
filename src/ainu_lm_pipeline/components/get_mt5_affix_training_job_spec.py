from kfp import dsl


@dsl.component(base_image="python:3.10")
def get_mt5_affix_training_job_spec(
    train_image_uri: str,
    push_to_hub: bool,
    dataset_revision: str,
) -> list:
    worker_pool_specs = [
        {
            "container_spec": {
                "image_uri": train_image_uri,
                "args": [
                    "mt5-affix",
                    "--dataset-name=aynumosir/ainu-corpora",
                    "--dataset-split=train",
                    f"--dataset-revision={dataset_revision}",
                    f"--push-to-hub={push_to_hub}",
                    "--num-train-epochs=10",
                    "--per-device-train-batch-size=32",
                    "--per-device-eval-batch-size=32",
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
