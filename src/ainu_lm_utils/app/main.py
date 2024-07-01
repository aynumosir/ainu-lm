from argparse import ArgumentParser

from .commands import generate


def main() -> None:
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    generate.add_parser(subparsers.add_parser("generate"))

    args = parser.parse_args()

    if args.command == "generate":
        generate.main(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")
