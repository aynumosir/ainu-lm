from pathlib import Path
from typing import Optional

from datasets import load_dataset
from google.cloud import aiplatform, storage
from google.cloud.storage import Blob

from ..trainers import RobertaTrainer, RobertaTrainerConfig


def language_model(
    output_dir: Path,
    logging_dir: Path,
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

    config = RobertaTrainerConfig(
        output_dir=output_dir,
        logging_dir=logging_dir,
        num_train_epochs=num_train_epochs,
        tokenizer_name_or_dir=tokenizer_dir,
        hypertune_enabled=hypertune_enabled,
    )

    trainer = RobertaTrainer(dataset, config=config)
    trainer.train()
