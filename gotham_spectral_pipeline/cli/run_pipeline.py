import argparse
import collections
import os
import pathlib
import pprint
import traceback

import loguru
import tqdm  # type: ignore

from .. import PositionSwitchedCalibration, SDFits, Spectrum, SpectrumAggregator, ZenithOpacity
from ..logging import capture_builtin_warnings, LogLimiter


def name() -> str:
    return __name__.split(".")[-1]


def help() -> str:
    return (
        "Run pipeline on a SDFits and generate a baseline-subtracted averaged spectrum."
    )


def configure_parser(parser: argparse.ArgumentParser):
    parser.add_argument("--sdfits", type=SDFits, required=True)
    parser.add_argument("--zenith_opacity", type=ZenithOpacity)
    parser.add_argument("--filter", type=str)
    parser.add_argument("--channel_width", type=float, required=True)
    parser.add_argument("--prefix", type=str)
    parser.add_argument("--output_directory", type=pathlib.Path, default="output")
    parser.add_argument("--logs_directory", type=pathlib.Path, default="logs")

    parser.add_argument("--Tsys_min_threshold", type=float, default=0.0)
    parser.add_argument("--Tsys_max_threshold", type=float, default="inf")
    parser.add_argument("--Tsys_min_success_rate", type=float, default=0.0)
    parser.add_argument("--Tsys_output_bad_session", type=bool, default=True)
    parser.add_argument(
        "--Tsys_bad_session_output_subdirectory", type=pathlib.Path, default="bad_Tsys"
    )


def main(args: argparse.Namespace):
    sdfits: SDFits = args.sdfits
    zenith_opacity: ZenithOpacity | None = args.zenith_opacity
    prefix = sdfits.path.name if args.prefix is None else args.prefix

    log_limiter = LogLimiter(prefix, args.logs_directory, WARNING=30.0)
    capture_builtin_warnings()

    if args.filter is None:
        filtered_rows = sdfits.rows
    else:
        filtered_rows = sdfits.rows.query(args.filter)
    paired_rows = PositionSwitchedCalibration.pair_up_rows(filtered_rows)

    if len(paired_rows) == 0:
        return

    Tsys_stats = dict(succeed=0, total=0)
    num_integration_dropped: dict[str, int] = collections.defaultdict(int)
    spectrum_aggregator = SpectrumAggregator(
        SpectrumAggregator.LinearTransformer(args.channel_width)
    )
    for paired_row in tqdm.tqdm(paired_rows):
        debug_indices = {
            f"{sigref},{calonoff}": int(paired_row[sigref][calonoff]["INDEX"])
            for sigref in paired_row
            for calonoff in paired_row[sigref]
        }
        try:
            sigrefpair = paired_row.get_paired_hdu(sdfits)
            if PositionSwitchedCalibration.should_be_discarded(sigrefpair):
                num_integration_dropped["Failed prechecks"] += 1
                continue

            (
                spectrum,
                spectrum_metadata,
            ) = PositionSwitchedCalibration.get_calibrated_spectrum(
                sigrefpair, freq_kwargs=dict(unit="Hz")
            )
            if spectrum is None:
                num_integration_dropped["No calibrated spectrum returned"] += 1
                continue

            Tsys_stats["total"] += 1
            if not spectrum_metadata["Tsys"] > args.Tsys_min_threshold:
                num_integration_dropped["Tsys exceeds min threshold"] += 1
                continue
            if not spectrum_metadata["Tsys"] < args.Tsys_max_threshold:
                num_integration_dropped["Tsys exceeds max threshold"] += 1
                continue
            Tsys_stats["succeed"] += 1

            assert spectrum.frequency is not None

            spectrum.flag_rfi()
            spectrum.flag_head_tail(nchannel=4096)

            assert spectrum.flag is not None
            rfi_in_body_count = (
                (
                    spectrum.flag[
                        (spectrum.flag & Spectrum.FlagReason.CHUNK_EDGES.value) == 0
                    ]
                    & Spectrum.FlagReason.RFI.value
                )
                != 0
            ).sum()
            if rfi_in_body_count > 256:
                loguru.logger.error(
                    f"Found {rfi_in_body_count} RFI channels while working on {sdfits.path = }, {debug_indices = }. This integration seems to be broken."
                )
                num_integration_dropped["Too many RFI channels"] += 1
                continue

            spectrum.flag_nan()
            is_signal = spectrum.detect_signal(nadjacent=128, alpha=1e-6)
            baseline_result = spectrum.fit_baseline(
                method="hybrid",
                polynomial_options=dict(max_degree=20),
                lomb_scargle_options=dict(
                    min_num_terms=0, max_num_terms=40, max_cycle=32
                ),
                mask=is_signal,
                residual_threshold=0.1,
            )
            if baseline_result is None:
                num_integration_dropped["Failed baseline fitting"] += 1
                continue

            baseline, _ = baseline_result
            baseline_subtracted_spectrum = Spectrum(
                intensity=spectrum.intensity - baseline(spectrum.frequency),
                frequency=spectrum.frequency,
                noise=spectrum.noise,
                flag=spectrum.flagged,
            )

            if zenith_opacity is not None:
                correction_factor = (
                    PositionSwitchedCalibration.get_temperature_correction_factor(
                        sigrefpair["sig"]["caloff"], zenith_opacity
                    )
                )
                if correction_factor is None:
                    num_integration_dropped[
                        "Can't retrieve opacity correction factor"
                    ] += 1
                    continue
                baseline_subtracted_spectrum *= correction_factor

            spectrum_aggregator.merge(baseline_subtracted_spectrum)
        except Exception:
            loguru.logger.critical(
                f"Uncaught exception while working on {sdfits.path = }, {debug_indices = }\n{traceback.format_exc()}"
            )
            num_integration_dropped["Uncaught expeption"] += 1
    if num_integration_dropped:
        loguru.logger.info(
            f"Some integrations dropped due to the following reasons:\n{pprint.pformat(dict(num_integration_dropped))}"
        )

    Tsys_success_rate = Tsys_stats["succeed"] / Tsys_stats["total"]
    if Tsys_success_rate < args.Tsys_min_success_rate:
        output_directory = (
            args.output_directory / args.Tsys_bad_session_output_subdirectory
        )
    else:
        output_directory = args.output_directory

    os.makedirs(output_directory, exist_ok=True)
    output_path = output_directory / prefix
    spectrum_aggregator.get_spectrum().to_npz(output_path)

    log_limiter.log_silence_report()
