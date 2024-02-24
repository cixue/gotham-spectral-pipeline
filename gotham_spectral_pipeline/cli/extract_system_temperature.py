import argparse
import pathlib

import pandas
import tqdm  # type: ignore

from .. import PositionSwitchedCalibration, SDFits
from ..logging import capture_builtin_warnings, LogLimiter


def name() -> str:
    return __name__.split(".")[-1]


def help() -> str:
    return "Extract system temperature from integration and relevant info and output to a csv file."


def return_exist_file(arg: str) -> pathlib.Path:
    path = pathlib.Path(arg)
    if path.exists():
        return path
    raise argparse.ArgumentTypeError(f"{arg} does not exist!")


def configure_parser(parser: argparse.ArgumentParser):
    parser.add_argument("--sdfits", type=return_exist_file, nargs="+", required=True)
    parser.add_argument("--filter", type=str)
    parser.add_argument("--output", type=argparse.FileType("w"), default="Tsys.csv")
    parser.add_argument("--prefix", type=str, default="Tsys")
    parser.add_argument("--logs_directory", type=pathlib.Path, default="logs")


def main(args: argparse.Namespace):
    log_limiter = LogLimiter(args.prefix, args.logs_directory, WARNING=30.0)
    capture_builtin_warnings()

    first_write = True
    for sdfits_path in tqdm.tqdm(args.sdfits):
        sdfits = SDFits(sdfits_path)
        if args.filter is None:
            filtered_rows = sdfits.rows
        else:
            filtered_rows = sdfits.rows.query(args.filter)
        paired_rows = PositionSwitchedCalibration.pair_up_rows(filtered_rows)

        batched_output: list[pandas.DataFrame] = list()
        for paired_row in tqdm.tqdm(paired_rows, leave=False):
            sigrefpair = paired_row.get_paired_hdu(sdfits)
            Tsys = PositionSwitchedCalibration.get_system_temperature(
                sigrefpair["ref"], trim_fraction=0.1
            )
            output = paired_row["ref"]["caloff"][[
                "PROJECT",
                "INDEX",
                "SCAN",
                "PLNUM",
                "IFNUM",
                "FEED",
                "FDNUM",
                "INT",
                "SAMPLER",
                "CENTFREQ",
                "BANDWIDTH",
                "EXPOSURE",
                "TSYS",
            ]].copy()
            output["TSYS"] = Tsys
            batched_output.append(output)
        pandas.concat(batched_output).to_csv(
            args.output, header=first_write, index=False
        )
        first_write = False
