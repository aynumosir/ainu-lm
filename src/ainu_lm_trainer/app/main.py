from .argument_parser import get_argument_parser
from .task_cache import cache
from .task_language_model import language_model
from .task_tokenizer import tokenizer

if __name__ == "__main__":
    argument_parser = get_argument_parser()
    args = argument_parser.parse_args()

    if args.task == "tokenizer":
        tokenizer(output_dir=args.output_dir, dataset_revision=args.dataset_revision)

    if args.task == "language-model":
        language_model(
            output_dir=args.output_dir,
            logging_dir=args.logging_dir,
            tokenizer_dir=args.tokenizer_dir,
            num_train_epochs=args.num_train_epochs,
            hypertune_enabled=args.hp_tune,
            dataset_revision=args.dataset_revision,
        )

    if args.task == "cache":
        cache()
