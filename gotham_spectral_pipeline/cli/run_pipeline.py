import argparse
import pathlib
import traceback

import loguru

from .. import PositionSwitchedCalibration, SDFits, Spectrum, SpectrumAggregator, ZenithOpacity


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
    parser.add_argument("--output_directory", type=pathlib.Path)


def main(args: argparse.Namespace):
    sdfits: SDFits = args.sdfits
    zenith_opacity: ZenithOpacity | None = args.zenith_opacity

    if args.filter is None:
        filtered_rows = sdfits.rows
    else:
        filtered_rows = sdfits.rows.query(args.filter)
    paired_rows = PositionSwitchedCalibration.pair_up_rows(filtered_rows)

    if len(paired_rows) == 0:
        return

    spectrum_aggregator = SpectrumAggregator(
        SpectrumAggregator.LinearTransformer(args.channel_width)
    )
    for paired_row in paired_rows:
        try:
            sigrefpair = paired_row.get_paired_hdu(sdfits)
            if PositionSwitchedCalibration.should_be_discarded(sigrefpair):
                continue

            spectrum = PositionSwitchedCalibration.get_calibrated_spectrum(
                sigrefpair, freq_kwargs=dict(unit="Hz")
            )
            if spectrum is None:
                continue

            assert spectrum.frequency is not None

            spectrum.flag_rfi()
            spectrum.flag_head_tail(nchannel=2048)
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
                    continue
                baseline_subtracted_spectrum *= correction_factor

            spectrum_aggregator.merge(baseline_subtracted_spectrum)
        except Exception:
            indices = {
                f"{sigref},{calonoff}": int(paired_row[sigref][calonoff]["INDEX"])
                for sigref in paired_row
                for calonoff in paired_row[sigref]
            }
            loguru.logger.critical(f"Uncaught exception while working on {sdfits.path = }, {indices = }\n{traceback.format_exc()}")

    output_directory: pathlib.Path = (
        pathlib.Path.cwd() if args.output_directory is None else args.output_directory
    )
    output_path = output_directory / sdfits.path.name
    spectrum_aggregator.get_spectrum().to_npz(output_path)
