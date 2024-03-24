from .argument_parser import get_argument_parser
from .commands import language_model, tokenizer

if __name__ == "__main__":
    argument_parser = get_argument_parser()
    args = argument_parser.parse_args()

    if args.model == "tokenizer":
        tokenizer(job_dir=args.job_dir)

    if args.model == "language_model":
        language_model(job_dir=args.job_dir, tokenizer_blob=args.tokenizer_blob)
