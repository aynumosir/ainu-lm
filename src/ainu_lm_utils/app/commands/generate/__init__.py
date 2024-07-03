from argparse import ArgumentParser, Namespace

from . import normalized, pos_tagged, round_trip_translations, rule_based_errors


def add_parser(parser: ArgumentParser) -> None:
    subparsers = parser.add_subparsers(dest="task")

    normalized.add_parser(subparsers.add_parser("normalized"))
    pos_tagged.add_parser(subparsers.add_parser("pos-tagged"))
    round_trip_translations.add_parser(subparsers.add_parser("round-trip-translations"))
    rule_based_errors.add_parser(subparsers.add_parser("rule-based-errors"))


def main(args: Namespace) -> None:
    if args.task == "normalized":
        normalized.main(args)
    elif args.task == "pos-tagged":
        pos_tagged.main(args)
    elif args.task == "round-trip-translations":
        round_trip_translations.main(args)
    elif args.task == "rule-based-errors":
        rule_based_errors.main(args)
    else:
        raise ValueError(f"Unknown task: {args.task}")
