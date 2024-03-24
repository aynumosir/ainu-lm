from datasets import load_dataset

from ..trainers import ByteLevelBPETokenizerTrainer
from .argument_parser import get_argument_parser

if __name__ == "__main__":
    argument_parser = get_argument_parser()
    args = argument_parser.parse_args()

    if args.model == "tokenizer":
        dataset = load_dataset("aynumosir/ainu-corpora", split="data")
        trainer = ByteLevelBPETokenizerTrainer(dataset, output_dir=args.output_dir)
        trainer.train()

    if args.model == "language_model":
        pass
