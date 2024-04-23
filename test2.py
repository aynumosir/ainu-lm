from transformers import T5TokenizerFast

tokenizer = T5TokenizerFast.from_pretrained("./models/sentencepiece")


encoding = tokenizer(
    "ne ne kusu nekon eci=iki yakka kamuy nukar akanak a=mutemus notakkaske eci=utari koyaytaraye eci=ki kun pe ne ruwe ne na” arid an pe hawki=an kane ohumse=an hine yay'okokokse'eciw=an kane ikirok pe a=kotetterke “hemka woy hu ohohoy”",
    text_target="ne wa ne kusu nekon eci=iki yakka kamuy nukar akanak a=mutemus notakkaske eci=utari koyaytaraye eci=ki kun pe ne ruwe ne na” ari an pe hawki=an kane ohumse=an hine yay'okokokse'eciw=an kane ikirok pe a=kotetterke “hemka woy hu ohohoy”",
    truncation=True,
    max_length=128,
    padding="max_length",
)


print(tokenizer.decode(encoding["input_ids"]))
print(tokenizer.decode(encoding["labels"]))
