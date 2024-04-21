from kfp import dsl


@dsl.component(base_image="python:3.10")
def get_mt5_gec_training_job_spec(
    train_image_uri: str,
    dataset_revision: str,
    # tokenizer_gcs_path: str,
) -> list:
    worker_pool_specs = [
        {
            "container_spec": {
                "image_uri": train_image_uri,
                "args": [
                    "mt5-gec",
                    "--num-train-epochs=10",
                    "--per-device-batch-size=64",
                    # f"--tokenizer-dir={tokenizer_gcs_path}",
                    f"--dataset-revision={dataset_revision}",
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
