from ....config import TaskPrefixType


def __make_ainu_language_identifier(
    example: dict, task_prefix_type: TaskPrefixType
) -> str:
    if task_prefix_type == TaskPrefixType.NONE:
        return "Ainu"

    if task_prefix_type == TaskPrefixType.DIALECT:
        if example["dialect"] is None:
            return "Ainu"
        else:
            return f"Ainu ({example['dialect']})"

    if task_prefix_type == TaskPrefixType.PRONOUN:
        return f"Ainu ({example['pronoun']})"

    if task_prefix_type == TaskPrefixType.ALL:
        if example["dialect"] is None:
            return f"Ainu ({example['pronoun']})"
        else:
            return f"Ainu ({example['dialect']}, {example['pronoun']})"

    raise ValueError(f"Unknown TaskPrefixType: {task_prefix_type}")


def ain2ja(example: dict, task_prefix_type: TaskPrefixType = TaskPrefixType.ALL) -> str:
    ainu = __make_ainu_language_identifier(example, task_prefix_type)
    return f"translate {ainu} to Japanese: "


def ja2ain(example: dict, task_prefix_type: TaskPrefixType = TaskPrefixType.ALL) -> str:
    ainu = __make_ainu_language_identifier(example, task_prefix_type)
    return f"translate Japanese to {ainu}: "
