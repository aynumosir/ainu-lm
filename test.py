import sys

from transformers import pipeline

generate = pipeline("text-generation", model="./models/gpt2")

result = generate(
    sys.argv[1],
    max_length=50,
    truncation=True,
    pad_token_id=25493,
    temperature=0.1,
    top_k=50,
    top_p=0.95,
    num_return_sequences=1,
)

print(result[0]["generated_text"])
