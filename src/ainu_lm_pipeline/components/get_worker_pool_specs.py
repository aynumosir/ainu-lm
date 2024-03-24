from kfp import dsl


@dsl.component(
    base_image="python:3.11",
    output_component_file="./pipelines/get_worker_pool_specs.yaml",
)
def get_worker_pool_specs(
    train_image_uri: str,
    tokenizer_gcs_path: str,
) -> list:
    worker_pool_specs = [
        {
            "container_spec": {
                "image_uri": train_image_uri,
                "args": [
                    "language-model",
                    "--hp-tune=True",
                    f"--tokenizer={tokenizer_gcs_path}",
                ],
            },
            "machine_spec": {
                "machine_type": "n1-standard-4",
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
