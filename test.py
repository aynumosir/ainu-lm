from transformers import T5TokenizerFast

# model = T5ForConditionalGeneration.from_pretrained("aynumosir/t5-base-ainu-gce")

tokenizer = T5TokenizerFast.from_pretrained("./test")
tokenizer.save_pretrained("./test")

# generate = pipeline(
#     "text2text-generation",
# )

# print(
#     generate(
#         "pirkare: inkarusi sekor ku=rehe an. kani sisam ku=ne korka aynu itak k=eyaypakasnu wa an na.",
#         max_length=100,
#         num_beams=4,
#         early_stopping=True,
#     )
# )
