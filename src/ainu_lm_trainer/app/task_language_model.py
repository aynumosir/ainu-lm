import os
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from google.cloud import aiplatform, storage
from google.cloud.storage import Blob

from ..models import JobDir
from ..trainers import RobertaTrainer, RobertaTrainerConfig


def language_model(
    job_dir: JobDir,
    tokenizer_blob: Blob,
    num_train_epochs: int,
    hypertune_enabled: Optional[bool] = None,
    tensorboard_id: Optional[str] = None,
    tensorboard_experiment_name: Optional[str] = None,
) -> None:
    aiplatform.init()

    client = storage.Client()
    dataset = load_dataset("aynumosir/ainu-corpora", split="data")
    dataset = dataset.map(lambda example: {"text": example["sentence"]})

    # Download tokenizer files
    tokenizer_dir = Path("/tmp/ainu-lm-trainer/tokenizer")
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    for blob in client.list_blobs(
        tokenizer_blob.bucket.name, prefix=tokenizer_blob.name
    ):
        filename = blob.name.split("/")[-1]
        blob.download_to_filename(str(tokenizer_dir / filename))

    # Create output directory
    output_dir = Path("/tmp/ainu-lm-trainer/lm")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = RobertaTrainerConfig(
        num_train_epochs=num_train_epochs,
        tokenizer_name_or_dir=tokenizer_dir,
        output_dir=output_dir,
        hypertune_enabled=hypertune_enabled,
        tensorboard_id=tensorboard_id,
        tensorboard_experiment_name=tensorboard_experiment_name,
    )

    trainer = RobertaTrainer(dataset, config=config)
    trainer.train()

    paths = [
        Path(os.path.join(output_dir, file))
        for file in os.listdir(output_dir)
        if os.path.isfile(os.path.join(output_dir, file))
    ]

    for path in paths:
        file = path.name
        blob = job_dir.resolve(file).to_blob(client=client)
        blob.upload_from_filename(path)
        print(f"Uploaded {file} to {str(job_dir)}")

    os.system(f"rm -rf {output_dir}")
