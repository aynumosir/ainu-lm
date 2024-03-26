from kfp import dsl


@dsl.component(
    base_image="python:3.10",
    output_component_file="./dist/get_lm_training_job_spec.yaml",
)
def get_lm_training_job_spec(
    train_image_uri: str,
    tokenizer_gcs_path: str,
    dataset_revision: str,
) -> list:
    worker_pool_specs = [
        {
            "container_spec": {
                "image_uri": train_image_uri,
                "args": [
                    "language-model",
                    "--num-train-epochs=1",
                    f"--tokenizer-dir={tokenizer_gcs_path}",
                    f"--dataset-revision={dataset_revision}",
                ],
            },
            # https://cloud.google.com/vertex-ai/docs/training/configure-compute?hl=ja#specifying_gpus
            "machine_spec": {
                "machine_type": "n1-standard-16",
                "accelerator_type": "NVIDIA_TESLA_V100",
                "accelerator_count": 2,
            },
            "replica_count": "1",
            "disk_spec": {
                "boot_disk_type": "pd-ssd",
                "boot_disk_size_gb": 100,
            },
        }
    ]

    return worker_pool_specs
