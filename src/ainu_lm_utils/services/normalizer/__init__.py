from . import normalizer_ainu, normalizer_japanese


def normalize(example: dict) -> dict:
    example["text"] = normalizer_ainu.normalize(example["text"])
    example = normalizer_japanese.normalize(example)
    return example
