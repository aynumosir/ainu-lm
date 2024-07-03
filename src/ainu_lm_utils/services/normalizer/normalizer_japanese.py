import re


# cspell: disable
def normalize_nabesawa_kamuyyukar2(text: str) -> str:
    matches = re.findall(r"【(.*?)】", text)
    return "".join(matches)


def normalize(example: dict) -> dict:
    if example["book"] != "鍋沢元蔵筆録ノート":
        return example

    example["text"] = normalize_nabesawa_kamuyyukar2(example["text"])
    return example
