from argparse import ArgumentParser, Namespace

from . import normalized


def add_parser(parser: ArgumentParser) -> None:
    subparsers = parser.add_subparsers(dest="task")

    normalized.add_parser(subparsers.add_parser("normalized"))


def main(args: Namespace) -> None:
    if args.task == "normalized":
        normalized.main(args)
    else:
        raise ValueError(f"Unknown task: {args.task}")
