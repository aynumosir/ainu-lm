import os

from prettytable import PrettyTable
from transformers import RobertaTokenizerFast, pipeline

tokenizer = RobertaTokenizerFast.from_pretrained(
    "aynumosir/roberta-base-ainu-pos", model_max_length=128
)

ner = pipeline(
    "ner",
    model="aynumosir/roberta-base-ainu-pos",
    tokenizer=tokenizer,
)

mapping = {
    "PPX": "人称接辞",
    "PSX": "人称接辞",
    "PF": "接頭辞",
    "N": "名詞",
    "NL": "位置名詞",
    "PRN": "代名詞",
    "NMLZ": "形式名詞",
    "COMP": "形式名詞",
    "PRP.N": "固有名詞",
    "VI": "自動詞",
    "VT": "他動詞",
    "VC": "完全動詞",
    "VD": "複他動詞",
    "AUX": "助動詞",
    "ADV": "副詞",
    "ADV.PP": "後置副詞",
    "DEM": "連体詞",
    "PP": "格助詞",
    "ADV.PRT": "副助詞",
    "ADV.CONJ": "副助詞",
    "CONJ": "接続助詞",
    "FIN.PRT": "終助詞",
    "NUM": "数詞",
    "N.INTERR": "疑問詞",
    "INTJ": "間投詞",
    "PUNCT": "記号類",
}


def merge_entities(entities):
    merged_entities = []
    for entity in entities:
        if not merged_entities:
            merged_entities.append(entity)
            continue

        prev_entity = merged_entities[-1]
        if prev_entity["entity"] == entity["entity"]:
            prev_entity["end"] = entity["end"]
        else:
            merged_entities.append(entity)

    return merged_entities


if __name__ == "__main__":
    text = os.sys.argv[1]

    entities = ner(text)
    entities = merge_entities(entities)

    results = []

    for entity in entities:
        span = text[entity["start"] : entity["end"]]
        entity_name = mapping.get(entity["entity"])
        results.append([entity_name, span])

    table = PrettyTable()
    table.field_names = ["品詞", "単語"]
    for result in results:
        table.add_row(result)

    print(table)
