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
    parser.add_argument("--exit_if_exist", action="store_true")

    parser.add_argument("--flag_head_tail_channel_number", type=int, default=4096)
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
    tsys_threshold_selector: TsysThresholdSelector | None = None
    if args.Tsys_dynamic_threshold:
        if args.Tsys_dynamic_threshold_selector == "LookupTable":
            tsys_threshold_selector = GbtTsysLookupTable()
        elif args.Tsys_dynamic_threshold_selector == "Hybrid":
            tsys_threshold_selector = GbtTsysHybridSelector()

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
            SpectrumAggregator.LinearTransformer(args.channel_width),
            options=dict(include_correlation=False),
        )
    )
    exposure_aggregator: dict[str, ExposureAggregator] = collections.defaultdict(
        lambda: ExposureAggregator(
            ExposureAggregator.LinearTransformer(args.channel_width)
        )
    )
    total_exposure_aggregator = ExposureAggregator(
        ExposureAggregator.LinearTransformer(args.channel_width)
    )
    Tsys_stats: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: dict(succeed=0, total=0)
    )

    def grouped_paired_rows_iter():
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
            for paired_row in grouped_paired_rows[group]:
                yield group, paired_row
                progress_bar.update(1)

    for group, paired_row in grouped_paired_rows_iter():
        debug_indices = {
            f"{sigref},{calonoff}": int(paired_row[sigref][calonoff]["INDEX"].iloc[0])
            for sigref in paired_row
            for calonoff in paired_row[sigref]
        }
        try:
            sigrefpair = paired_row.get_paired_hdu(sdfits)
            exposure = PositionSwitchedCalibration.get_exposure(sigrefpair["sig"])
            if exposure is None:
                num_integration_dropped["No exposure returned"] += 1
                continue
            total_exposure_aggregator.merge(exposure)

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

            Tsys_min_threshold = args.Tsys_min_threshold
            Tsys_max_threshold = args.Tsys_max_threshold
            if args.Tsys_dynamic_threshold:
                assert tsys_threshold_selector is not None
                central_frequency = sigrefpair["sig"]["caloff"][0].header["OBSFREQ"]
                dynamic_threshold = tsys_threshold_selector.get_threshold(
                    central_frequency
                )
                Tsys_min_threshold *= dynamic_threshold
                Tsys_max_threshold *= dynamic_threshold

            Tsys_stats[group]["total"] += 1
            if not spectrum_metadata["Tsys"] > Tsys_min_threshold:
                num_integration_dropped["Tsys exceeds min threshold"] += 1
                continue
            if not spectrum_metadata["Tsys"] < Tsys_max_threshold:
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
                    ~spectrum.flagged(Spectrum.FlagReason.CHUNK_EDGES)
                    & spectrum.flagged(
                        Spectrum.FlagReason.FREQUENCY_DOMAIN_RFI
                        | Spectrum.FlagReason.TIME_DOMAIN_RFI
                    )
                ).sum()
                if rfi_in_body_count > args.max_rfi_channel:
                    loguru.logger.error(
                        f"Found {rfi_in_body_count} RFI channels while working on {sdfits.path = }, {debug_indices = }. This integration seems to be broken."
                    )
                    num_integration_dropped["Too many RFI channels"] += 1
                    continue

            spectrum.flag_nan()
            spectrum.flag_signal()
            baseline_result = spectrum.fit_baseline(
                method="hybrid",
                polynomial_options=dict(max_degree=20),
                lomb_scargle_options=dict(
                    min_num_terms=0, max_num_terms=40, max_cycle=32
                ),
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
                    raise Halt("Opacity temperature correction enabled but failed.")
                baseline_subtracted_spectrum *= correction_factor

            baseline_subtracted_spectrum.flag_valid_data()
            exposure.exposure[
                ~baseline_subtracted_spectrum.flagged(Spectrum.FlagReason.VALID_DATA)
            ] = 0.0

            spectrum_aggregator[group].merge(baseline_subtracted_spectrum)
            exposure_aggregator[group].merge(exposure)
        except Halt as e:
            loguru.logger.critical(*e.args)
            tqdm.tqdm.write(*e.args)
            sys.exit(1)
        except Exception:
            loguru.logger.critical(
                f"Uncaught exception while working on {sdfits.path = }, {debug_indices = }\n{traceback.format_exc()}"
            )
            num_integration_dropped["Uncaught expeption"] += 1
    if num_integration_dropped:
        loguru.logger.info(
            f"Some integrations dropped due to the following reasons:\n{pprint.pformat(dict(num_integration_dropped))}"
        )

    final_spectrum_aggregator = SpectrumAggregator(
        SpectrumAggregator.LinearTransformer(args.channel_width),
        options=dict(include_correlation=True),
    )
    final_exposure_aggregator = ExposureAggregator(
        ExposureAggregator.LinearTransformer(args.channel_width)
    )
    for group in sorted(Tsys_stats.keys()):
        if Tsys_stats[group]["total"] == 0:
            Tsys_success_rate = 0.0
        else:
            Tsys_success_rate = (
                Tsys_stats[group]["succeed"] / Tsys_stats[group]["total"]
            )
        if Tsys_success_rate >= args.Tsys_min_success_rate:
            if group in spectrum_aggregator:
                final_spectrum_aggregator.merge(spectrum_aggregator[group])
            if group in exposure_aggregator:
                final_exposure_aggregator.merge(exposure_aggregator[group])
        else:
            loguru.logger.info(
                f"Dropped integrations in {group = } since success rate = {Tsys_success_rate} < required minimum success rate = {args.Tsys_min_success_rate}"
            )

    output_directory = args.output_directory
    os.makedirs(output_directory, exist_ok=True)

    if final_spectrum_aggregator is not None:
        final_spectrum_aggregator.get_spectrum().to_npz(output_directory / prefix)

    total_exposure = total_exposure_aggregator.get_spectrum()
    total_exposure.to_npz(output_directory / (prefix + ".total_exposure"))

    exposure = total_exposure
    exposure.exposure[:] = 0.0
    del total_exposure

    if final_exposure_aggregator is not None:
        final_exposure_aggregator.merge(exposure)
        exposure = final_exposure_aggregator.get_spectrum()
    exposure.to_npz(output_directory / (prefix + ".exposure"))

    log_limiter.log_silence_report()
