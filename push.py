from transformers import T5ForConditionalGeneration, T5TokenizerFast

MODEL_NAME = "aynumosir/t5-base-ainu-gce"

tokenizer = T5TokenizerFast.from_pretrained("./models/sentencepiece")
model = T5ForConditionalGeneration.from_pretrained("./models/t5-gce")

tokenizer.push_to_hub(MODEL_NAME)
model.push_to_hub(MODEL_NAME)
