from kfp import dsl


@dsl.component(
    base_image="python:3.10",
    output_component_file="./dist/get_tokenizer_training_job_spec.yaml",
)
def get_tokenizer_training_job_spec(
    train_image_uri: str,
) -> list:
    worker_pool_specs = [
        {
            "container_spec": {
                "image_uri": train_image_uri,
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
    ]

    return worker_pool_specs
