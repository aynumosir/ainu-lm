from typing import Optional

from kfp import dsl


@dsl.component(base_image="python:3.10")
def get_mt_ain2ja_training_job_spec(
    train_image_uri: str,
    dataset_name: str,
    dataset_revision: str,
    base_model: str,
    hub_model_id: str,
    push_to_hub: bool,
    base_tokenizer: Optional[str] = None,
) -> list:
    worker_pool_specs = [
        {
            "container_spec": {
                "image_uri": train_image_uri,
                "args": [
                    "train",
                    "mt-ain2ja",
                    f"--base-model={base_model}",
                    f"--dataset-name={dataset_name}",
                    f"--dataset-revision={dataset_revision}",
                    "--num-train-epochs=20",
                    "--per-device-train-batch-size=32",
                    "--per-device-eval-batch-size=32",
                    "--learning-rate=6e-4",
                    "--weight-decay=0.01",
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

    if base_tokenizer:
        worker_pool_specs[0]["container_spec"]["args"].append(
            f"--base-tokenizer={base_tokenizer}"
        )

    return worker_pool_specs
