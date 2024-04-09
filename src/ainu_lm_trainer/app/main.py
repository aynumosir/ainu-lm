from .argument_parser import get_argument_parser
from .task_byte_level_bpe import byte_level_bpe
from .task_gpt2 import gpt2
from .task_roberta import roberta

if __name__ == "__main__":
    argument_parser = get_argument_parser()
    args = argument_parser.parse_args()

    if args.task == "byte-level-bpe":
        byte_level_bpe(
            output_dir=args.output_dir, dataset_revision=args.dataset_revision
        )

    if args.task == "roberta":
        roberta(
            model_dir=args.model_dir,
            checkpoint_dir=args.checkpoint_dir,
            logging_dir=args.logging_dir,
            tokenizer_dir=args.tokenizer_dir,
            num_train_epochs=args.num_train_epochs,
            dataset_revision=args.dataset_revision,
        )

    if args.task == "gpt2":
        gpt2(
            model_dir=args.model_dir,
            checkpoint_dir=args.checkpoint_dir,
            logging_dir=args.logging_dir,
            tokenizer_dir=args.tokenizer_dir,
            num_train_epochs=args.num_train_epochs,
            dataset_revision=args.dataset_revision,
        )
