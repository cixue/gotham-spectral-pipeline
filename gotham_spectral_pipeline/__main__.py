import argparse
import sys

from .cli import extract_system_temperature, generate_cleo_command, run_pipeline


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
        extract_system_temperature,
        generate_cleo_command,
        run_pipeline,
    ]
    entry_points = dict()
    for subcommand in subcommands:
        subparser = subparsers.add_parser(subcommand.name(), help=subcommand.help())
        subcommand.configure_parser(subparser)
        entry_points[subcommand.name()] = subcommand.main

    args = parser.parse_args()
    exit_code = entry_points[args.command](args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
