from datasets import load_dataset


def cache() -> None:
    load_dataset("aynumosir/ainu-corpora")
    print("Prefetching complete!")
