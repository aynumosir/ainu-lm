from transformers import T5ForConditionalGeneration, T5TokenizerFast

MODEL_NAME = "aynumosir/t5-base-ainu-gce"

tokenizer = T5TokenizerFast.from_pretrained("./models/sentencepiece")
model = T5ForConditionalGeneration.from_pretrained("./checkpoints/checkpoint-3500/")

TASK_PREFIX = "pirkare: "
SENTENCE = "oya mosir un itak a=epakasnu hu etoko ta, Esuperanto eraman yak pirka sekor ye utar ka oka."

input_ids = tokenizer(TASK_PREFIX + SENTENCE, return_tensors="pt")["input_ids"]
outputs = model.generate(input_ids)


print(
    tokenizer.decode(
        outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
    ),
)
