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


def configure_parser(parser: argparse.ArgumentParser):
    parser.add_argument("--sdfits", type=SDFits, nargs="+", required=True)
    parser.add_argument("--filter", type=str)
    parser.add_argument("--output", type=argparse.FileType("w"), default="Tsys.csv")
    parser.add_argument("--prefix", type=str, default="Tsys")
    parser.add_argument("--logs_directory", type=pathlib.Path, default="logs")


def main(args: argparse.Namespace):
    log_limiter = LogLimiter(args.prefix, args.logs_directory, WARNING=30.0)
    capture_builtin_warnings()

    first_write = True
    for sdfits in tqdm.tqdm(args.sdfits):
        if args.filter is None:
            filtered_rows = sdfits.rows
        else:
            filtered_rows = sdfits.rows.query(args.filter)
        paired_rows = PositionSwitchedCalibration.pair_up_rows(filtered_rows)

        batched_output: list[pandas.DataFrame] = list()
        for paired_row in tqdm.tqdm(paired_rows, leave=False):
            sigrefpair = paired_row.get_paired_hdu(sdfits)
            if PositionSwitchedCalibration.should_be_discarded(sigrefpair):
                continue
            Tsys = PositionSwitchedCalibration.get_system_temperature(
                sigrefpair["ref"], trim_fraction=0.1
            )
            output = paired_row["ref"]["caloff"][
                ["PROJECT", "FDNUM", "PLNUM", "CENTFREQ", "BANDWIDTH", "TSYS"]
            ].copy()
            output["TSYS"] = Tsys
            batched_output.append(output)
        pandas.concat(batched_output).to_csv(
            args.output, header=first_write, index=False
        )
        first_write = False
