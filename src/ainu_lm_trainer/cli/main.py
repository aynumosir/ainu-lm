from pathlib import Path

from datasets import load_dataset

from ..trainers import BpeTrainer

if __name__ == "__main__":
    dataset = load_dataset("aynumosir/ainu-corpora")

    corpus = "\n".join(dataset["data"]["sentence"])

    with open("/tmp/corpus.txt", "w") as f:
        f.write(corpus)

    trainer = BpeTrainer(
        corpus_file=Path("/tmp/corpus.txt"), output_dir=Path("./models/tokenizer")
    )

    result = trainer.train()

    # print(result.model_path)
