from argparse import ArgumentParser, Namespace

from . import kana, normalized


def add_parser(parser: ArgumentParser) -> None:
    subparsers = parser.add_subparsers(dest="task")

    normalized.add_parser(subparsers.add_parser("normalized"))
    kana.add_parser(subparsers.add_parser("kana"))


def main(args: Namespace) -> None:
    if args.task == "normalized":
        normalized.main(args)
    elif args.task == "kana":
        kana.main(args)
    else:
        raise ValueError(f"Unknown task: {args.task}")
