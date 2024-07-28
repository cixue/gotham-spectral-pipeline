import argparse
import collections
import os
import pathlib
import pprint
import sys
import traceback

import loguru
import tqdm  # type: ignore

from .. import __version__
from .. import Spectrum, SpectrumAggregator
from .. import Exposure, ExposureAggregator
from .. import PositionSwitchedCalibration, SDFits, ZenithOpacity
from .. import GbtTsysLookupTable, GbtTsysHybridSelector, TsysThresholdSelector
from .. import SigRefPairedRows
from .. import Pipeline
from ..logging import capture_builtin_warnings, LogLimiter


def name() -> str:
    return __name__.split(".")[-1]


def help() -> str:
    return (
        "Run pipeline on a SDFits and generate a baseline-subtracted averaged spectrum."
    )


def configure_parser(parser: argparse.ArgumentParser):
    parser.add_argument("--sdfits", type=pathlib.Path, required=True)
    parser.add_argument("--zenith_opacity", type=pathlib.Path)
    parser.add_argument("--filter", type=str)
    parser.add_argument("--prefix", type=str)
    parser.add_argument("--output_directory", type=pathlib.Path, default="output")
    parser.add_argument("--logs_directory", type=pathlib.Path, default="logs")
    parser.add_argument("--exit_if_exist", action="store_true")

    parser.add_argument("--channel_width", type=float, required=True)
    parser.add_argument("--flag_head_tail_channel_number", type=int, default=1024)
    parser.add_argument("--max_rfi_channel", type=int, default=256)

    parser.add_argument("--Tsys_dynamic_threshold", action="store_true")
    parser.add_argument(
        "--Tsys_dynamic_threshold_selector",
        type=str,
        default="LookupTable",
        choices=["LookupTable", "Hybrid"],
    )
    parser.add_argument(
        "--Tsys_min_threshold",
        type=float,
        default=0.0,
        help="If Tsys_dynamic_threshold is not enabled, it represents the absolute value used for the minimum threshold. If Tsys_dynamic_threshold is enabled, it represents the multiplier for the minimum threshold.",
    )
    parser.add_argument(
        "--Tsys_max_threshold",
        type=float,
        default="inf",
        help="If Tsys_dynamic_threshold is not enabled, it represents the absolute value used for the maximum threshold. If Tsys_dynamic_threshold is enabled, it represents the multiplier for the maximum threshold.",
    )
    parser.add_argument("--Tsys_min_success_rate", type=float, default=0.0)


class Halt(RuntimeError):
    pass


def main(args: argparse.Namespace):
    prefix = args.sdfits.name if args.prefix is None else args.prefix

    if args.exit_if_exist and (args.output_directory / (prefix + ".npz")).exists():
        return

    log_limiter = LogLimiter(prefix, args.logs_directory, WARNING=30.0)
    capture_builtin_warnings()

    loguru.logger.info(f"Current version: {__version__}")

    quoted_arguments = [
        f"'{argument}'" if " " in argument else argument for argument in sys.argv
    ]
    loguru.logger.info(
        f"Pipeline started with the following options:\n{pprint.pformat(vars(args), sort_dicts=False)}\n"
        f"with the following command:\n{' '.join(quoted_arguments)}"
    )

    sdfits: SDFits = SDFits(args.sdfits)
    zenith_opacity: ZenithOpacity | None = (
        None if args.zenith_opacity is None else ZenithOpacity(args.zenith_opacity)
    )

    if args.filter is None:
        filtered_rows = sdfits.rows
    else:
        filtered_rows = sdfits.rows.query(args.filter)
    paired_rows = PositionSwitchedCalibration.pair_up_rows(filtered_rows)

    if len(paired_rows) == 0:
        return

    def grouped_paired_rows_iter(paired_rows: list[SigRefPairedRows]):
        grouped_paired_rows = collections.defaultdict(list)
        for paired_row in paired_rows:
            group = paired_row.metadata["group"]["sampler"]
            grouped_paired_rows[group].append(paired_row)

        progress_bar = tqdm.tqdm(
            total=len(paired_rows),
            dynamic_ncols=True,
            smoothing=0.0,
        )

        for group in grouped_paired_rows:
            yield group, grouped_paired_rows[group]
            progress_bar.update(len(grouped_paired_rows[group]))

    outputs: dict[str, Pipeline.Output] = dict()
    for group, paired_rows in grouped_paired_rows_iter(paired_rows):
        output = (
            Pipeline(
                sdfits,
                zenith_opacity,
                paired_rows,
                Pipeline.Options.from_namespace(args),
            )
            .calibrate()
            .get_output()
        )
        outputs[group] = output
        if not output.success:
            loguru.logger.info(
                f"Pipeline failed while calibrating {group = } due to the following reason:\n{output.reason}"
            )

    integration_dropped_reason = collections.Counter[str]()
    for output in outputs.values():
        if output.success:
            integration_dropped_reason += output.integration_dropped_reason
    if integration_dropped_reason:
        loguru.logger.info(
            f"Some integrations dropped due to the following reasons:\n{pprint.pformat(dict(integration_dropped_reason))}"
        )

    final_total_exposure = (
        ExposureAggregator(ExposureAggregator.LinearTransformer(args.channel_width))
        .merge_all(output.total_exposure for output in outputs.values())
        .get_spectrum()
    )
    final_exposure = (
        ExposureAggregator(ExposureAggregator.LinearTransformer(args.channel_width))
        .merge(final_total_exposure, init_only=True)
        .merge_all(output.exposure for output in outputs.values() if output.success)
        .get_spectrum()
    )
    final_spectrum = (
        SpectrumAggregator(
            SpectrumAggregator.LinearTransformer(args.channel_width),
            options=dict(include_correlation=True),
        )
        .merge_all(output.spectrum for output in outputs.values() if output.success)
        .get_spectrum()
    )

    output_directory = args.output_directory
    os.makedirs(output_directory, exist_ok=True)

    if any(output.success for output in outputs.values()):
        final_spectrum.to_npz(output_directory / prefix)

    final_total_exposure.to_npz(output_directory / (prefix + ".total_exposure"))
    final_exposure.to_npz(output_directory / (prefix + ".exposure"))

    log_limiter.log_silence_report()
