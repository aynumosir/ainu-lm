from argparse import ArgumentParser

from .commands import compile, push, schedule, submit


def main() -> None:
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    compile.add_parser(subparsers.add_parser("compile"))
    submit.add_parser(subparsers.add_parser("submit"))
    push.add_parser(subparsers.add_parser("push"))
    schedule.add_parser(subparsers.add_parser("schedule"))

    args = parser.parse_args()

    if args.command == "compile":
        compile.main(args)
    elif args.command == "submit":
        submit.main(args)
    elif args.command == "push":
        push.main(args)
    elif args.command == "schedule":
        schedule.main(args)
