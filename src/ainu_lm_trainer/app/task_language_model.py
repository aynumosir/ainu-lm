import os
from pathlib import Path

from datasets import load_dataset
from google.cloud import storage
from google.cloud.storage import Blob

from ...models import JobDir
from ...trainers import RobertaTrainer


def language_model(job_dir: JobDir, tokenizer_blob: Blob) -> None:
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
    trainer = RobertaTrainer(
        dataset, tokenizer_name_or_dir=tokenizer_dir, output_dir=output_dir
    )
    trainer.train()

    paths = [
        Path(os.path.join(output_dir, file))
        for file in os.listdir(output_dir)
        if os.path.isfile(os.path.join(output_dir, file))
    ]

    for path in paths:
        file = path.name
        blob = job_dir.resolve(file).blob
        blob.upload_from_filename(path)
        print(f"Uploaded {file} to {job_dir.blob.bucket.name}/{job_dir.blob.name}")

    os.system(f"rm -rf {output_dir}")
