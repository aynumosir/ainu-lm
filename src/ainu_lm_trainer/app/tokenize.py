from transformers import RobertaTokenizerFast

save_path = "./models/tokenizer/tokenizer.json"

if __name__ == "__main__":
    tokenizer = RobertaTokenizerFast.from_pretrained("./models/tokenizer")

    test_input = "Kánto or wa yaku sak no a=ranke p sinep ka isam"
    print(tokenizer.tokenize(test_input))
    print(tokenizer.encode(test_input))
    print(tokenizer.decode(tokenizer.encode(test_input)))
