from transformers import T5TokenizerFast

if __name__ == "__main__":
    tokenizer = T5TokenizerFast(
        vocab_file="./models/tokenizer/tokenizer.model", legacy=False
    )

    test_input = "Kanto or wa yaku sak no a=ranke p sinep ka isam"

    print(tokenizer.tokenize(test_input))
    print(tokenizer.encode(test_input))
    print(tokenizer.decode(tokenizer.encode(test_input)))
