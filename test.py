from datasets import Dataset

ds = Dataset.from_list(
    [
        {"text": "This is a test."},
        {"text": "This is another test."},
        {"text": "This is yet another test."},
    ]
)

# convert "text" to "sentence"
ds2 = ds.map(lambda example: {"sentence": example["text"]})

print(type(ds))
