import os
from pathlib import Path

from datasets import load_dataset

from ...models import JobDir
from ...trainers import ByteLevelBPETokenizerTrainer


def tokenizer(job_dir: JobDir) -> None:
    dataset = load_dataset("aynumosir/ainu-corpora", split="data")

    output_dir = Path("/tmp/ainu-lm-trainer/tokenizer")
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = ByteLevelBPETokenizerTrainer(dataset, output_dir=output_dir)
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
