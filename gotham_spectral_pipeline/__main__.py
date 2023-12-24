import argparse
import sys

from .cli import generate_cleo_command, run_pipeline
from .logging import capture_builtin_warnings, LogLimiter


def main():
    parser = argparse.ArgumentParser(
        prog=f"python -m {__package__}",
        description="Gotham Spectral Pipeline",
    )
    subparsers = parser.add_subparsers(
        title="commands",
        description="The following subcommands are available.",
        dest="command",
        required=True,
        metavar="COMMAND",
    )

    subcommands = [
        generate_cleo_command,
        run_pipeline,
    ]
    entry_points = dict()
    for subcommand in subcommands:
        subparser = subparsers.add_parser(subcommand.name(), help=subcommand.help())
        subcommand.configure_parser(subparser)
        entry_points[subcommand.name()] = subcommand.main

    capture_builtin_warnings()
    log_limiter = LogLimiter(WARNING=30.0)

    args = parser.parse_args()
    exit_code = entry_points[args.command](args)
    log_limiter.log_silence_report()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
