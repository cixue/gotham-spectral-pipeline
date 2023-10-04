import argparse

from .. import SDFits, ZenithOpacity


def name() -> str:
    return __name__.split(".")[-1]


def help() -> str:
    return "Generate CLEO command for getting opacity table for a SDFits file."


def configure_parser(parser: argparse.ArgumentParser):
    parser.add_argument("sdfits", type=SDFits)


def main(args: argparse.Namespace):
    command = ZenithOpacity.generate_cleo_command(
        (min(args.sdfits.rows.DATEOBS), max(args.sdfits.rows.DATEOBS)),
        frequency_range=(1.0, 50.0),
        force_remove=True,
    )
    print(command)
