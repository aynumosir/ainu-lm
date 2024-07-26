def ain2ja(example: dict) -> str:
    if example["dialect"] is not None:
        return (
            f"translate Ainu ({example['dialect']}, {example['pronoun']}) to Japanese: "
        )
    else:
        return f"translate Ainu (沙流, {example['pronoun']}) to Japanese: "


def ja2ain(example: dict) -> str:
    if example["dialect"] is not None:
        return (
            f"translate Japanese to Ainu ({example['dialect']}, {example['pronoun']}): "
        )
    else:
        return f"translate Japanese to Ainu (沙流, {example['pronoun']}): "
