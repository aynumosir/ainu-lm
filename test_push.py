from transformers import RobertaForTokenClassification, RobertaTokenizerFast

model = RobertaForTokenClassification.from_pretrained("./roberta-base-ainu-pos/")
tokenizer = RobertaTokenizerFast.from_pretrained("./roberta-base-ainu-pos/")

model.push_to_hub("aynumosir/roberta-base-ainu-pos")
tokenizer.push_to_hub("aynumosir/roberta-base-ainu-pos")
