from argparse import ArgumentParser

from .commands import train


def main() -> None:
    parser = ArgumentParser()

    subparsers = parser.add_subparsers(dest="subcommand")
    train.add_parser(subparsers.add_parser("train"))

    args = parser.parse_args()

    if args.subcommand == "train":
        train.main(args)
    else:
        raise ValueError(f"Invalid subcommand: {args.subcommand}")
