from transformers import T5ForConditionalGeneration, T5TokenizerFast

tokenizer = T5TokenizerFast.from_pretrained("./models/sentencepiece")
model = T5ForConditionalGeneration.from_pretrained("./models/t5-gce/")


def correct(text: str) -> str:
    input_text = f"pirkare: {text}"
    inputs = tokenizer.encode(
        input_text,
        return_tensors="pt",
        max_length=128,
        padding="max_length",
        truncation=True,
    )
    corrected_ids = model.generate(
        inputs, max_length=128, num_beams=5, early_stopping=True
    )
    corrected_sentence = tokenizer.decode(corrected_ids[0], skip_special_tokens=True)
    return corrected_sentence


print(correct("Puraha or ta ci=ye itak."))
