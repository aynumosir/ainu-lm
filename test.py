from transformers import MT5ForConditionalGeneration, T5TokenizerFast

model = MT5ForConditionalGeneration.from_pretrained("./models/mt5-gec")
tokenizer = T5TokenizerFast.from_pretrained("google/mt5-small")

MODEL_NAME = "aynumosir/mt5-small-ainu-gec"

model.push_to_hub(MODEL_NAME)
tokenizer.push_to_hub(MODEL_NAME)
