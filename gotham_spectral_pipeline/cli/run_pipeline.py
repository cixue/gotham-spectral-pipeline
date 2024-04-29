import argparse
import collections
import os
import pathlib
import pprint
import traceback
import typing

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
    parser.add_argument("--sdfits", type=pathlib.Path, required=True)
    parser.add_argument("--zenith_opacity", type=pathlib.Path)
    parser.add_argument("--filter", type=str)
    parser.add_argument("--channel_width", type=float, required=True)
    parser.add_argument("--prefix", type=str)
    parser.add_argument("--output_directory", type=pathlib.Path, default="output")
    parser.add_argument("--logs_directory", type=pathlib.Path, default="logs")
    parser.add_argument("--grouped_by_sampler", action="store_true")
    parser.add_argument("--exit_if_exist", action="store_true")

    parser.add_argument("--flag_head_tail_channel_number", type=int, default=4096)
    parser.add_argument("--max_rfi_channel", type=int, default=256)

    parser.add_argument("--Tsys_min_threshold", type=float, default=0.0)
    parser.add_argument("--Tsys_max_threshold", type=float, default="inf")
    parser.add_argument("--Tsys_min_success_rate", type=float, default=0.0)


def main(args: argparse.Namespace):
    prefix = args.sdfits.name if args.prefix is None else args.prefix

    if args.exit_if_exist and (args.output_directory / (prefix + ".npz")).exists():
        return

    sdfits: SDFits = SDFits(args.sdfits)
    zenith_opacity: ZenithOpacity | None = (
        None if args.zenith_opacity is None else ZenithOpacity(args.zenith_opacity)
    )

    log_limiter = LogLimiter(prefix, args.logs_directory, WARNING=30.0)
    capture_builtin_warnings()

    if args.filter is None:
        filtered_rows = sdfits.rows
    else:
        filtered_rows = sdfits.rows.query(args.filter)
    paired_rows = PositionSwitchedCalibration.pair_up_rows(filtered_rows)

    if len(paired_rows) == 0:
        return

    num_integration_dropped: dict[str, int] = collections.defaultdict(int)
    spectrum_aggregator: dict[str, SpectrumAggregator] = collections.defaultdict(
        lambda: SpectrumAggregator(
            SpectrumAggregator.LinearTransformer(args.channel_width)
        )
    )
    Tsys_stats: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: dict(succeed=0, total=0)
    )
    all_groups: set[str] = set()
    for paired_row in tqdm.tqdm(paired_rows):
        if args.grouped_by_sampler:
            group = paired_row.metadata["group"]["sampler"]
        else:
            group = "none"
        all_groups.add(group)

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
                sigrefpair, freq_kwargs=dict(unit="Hz"), return_metadata=True
            )
            if spectrum is None:
                num_integration_dropped["No calibrated spectrum returned"] += 1
                continue

            Tsys_stats[group]["total"] += 1
            if not spectrum_metadata["Tsys"] > args.Tsys_min_threshold:
                num_integration_dropped["Tsys exceeds min threshold"] += 1
                continue
            if not spectrum_metadata["Tsys"] < args.Tsys_max_threshold:
                num_integration_dropped["Tsys exceeds max threshold"] += 1
                continue
            Tsys_stats[group]["succeed"] += 1

            assert spectrum.frequency is not None

            spectrum.flag_time_domain_rfi(spectrum_metadata)
            spectrum.flag_frequency_domain_rfi()
            spectrum.flag_head_tail(nchannel=args.flag_head_tail_channel_number)

            assert spectrum.flag is not None
            if args.max_rfi_channel > 0:
                rfi_in_body_count = (
                    (
                        spectrum.flag[
                            (spectrum.flag & Spectrum.FlagReason.CHUNK_EDGES.value) == 0
                        ]
                        & (
                            Spectrum.FlagReason.FREQUENCY_DOMAIN_RFI.value
                            | Spectrum.FlagReason.TIME_DOMAIN_RFI.value
                        )
                    )
                    != 0
                ).sum()
                if rfi_in_body_count > args.max_rfi_channel:
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
                flag=spectrum.flag,
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

            spectrum_aggregator[group].merge(baseline_subtracted_spectrum)
        except Exception:
            loguru.logger.critical(
                f"Uncaught exception while working on {sdfits.path = }, {debug_indices = }\n{traceback.format_exc()}"
            )
            num_integration_dropped["Uncaught expeption"] += 1
    if num_integration_dropped:
        loguru.logger.info(
            f"Some integrations dropped due to the following reasons:\n{pprint.pformat(dict(num_integration_dropped))}"
        )

    final_spectrum_aggregator: SpectrumAggregator | None = None
    for group in sorted(all_groups):
        Tsys_success_rate = Tsys_stats[group]["succeed"] / Tsys_stats[group]["total"]
        if Tsys_success_rate >= args.Tsys_min_success_rate:
            if final_spectrum_aggregator is None:
                final_spectrum_aggregator = spectrum_aggregator[group]
            else:
                final_spectrum_aggregator.merge(spectrum_aggregator[group])
        else:
            loguru.logger.info(
                f"Dropped integrations in {group = } since success rate = {Tsys_success_rate} < required minimum success rate = {args.Tsys_min_success_rate}"
            )

    if final_spectrum_aggregator is not None:
        output_directory = args.output_directory
        os.makedirs(output_directory, exist_ok=True)
        output_path = output_directory / prefix
        final_spectrum_aggregator.get_spectrum().to_npz(output_path)

    log_limiter.log_silence_report()
