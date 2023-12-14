import argparse
import pathlib

import numpy

from .. import PairedHDU, PositionSwitchedCalibration, SDFits, Spectrum, ZenithOpacity


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
    parser.add_argument("--start_frequency", type=float, required=True)
    parser.add_argument("--stop_frequency", type=float, required=True)
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

    start_frequency: float = args.start_frequency
    stop_frequency: float = args.stop_frequency
    channel_width: float = args.channel_width
    bandwidth = stop_frequency - start_frequency
    nchannel = int(bandwidth // channel_width)
    frequency = start_frequency + numpy.arange(nchannel) * channel_width
    weighted_intensity = numpy.zeros(nchannel)
    inverse_variance = numpy.zeros(nchannel)

    for paired_row in paired_rows:
        paired_hdu = PairedHDU.from_paired_row(sdfits, paired_row)
        if paired_hdu.should_be_discarded():
            continue

        spectrum = PositionSwitchedCalibration.get_calibrated_spectrum(
            paired_hdu, freq_kwargs=dict(unit="Hz")
        )
        if spectrum is None:
            continue

        spectrum.flag_rfi()
        spectrum.flag_head_tail(nchannel=2048)
        is_signal = spectrum.detect_signal(nadjacent=128, alpha=1e-6)
        baseline_result = spectrum.fit_baseline(
            method="hybrid",
            polynomial_options=dict(max_degree=20),
            lomb_scargle_options=dict(min_num_terms=0, max_num_terms=40, max_cycle=32),
            mask=is_signal,
            residual_threshold=0.1,
        )
        if baseline_result is None:
            continue

        baseline, baseline_info = baseline_result
        baseline_subtracted_spectrum = Spectrum(
            spectrum.frequency,
            spectrum.intensity - baseline(spectrum.frequency),
            spectrum.noise,
        )

        assert baseline_subtracted_spectrum.noise is not None

        baseline_subtracted_spectrum.noise[spectrum.flagged] = numpy.inf

        if zenith_opacity is not None:
            correction_factor = (
                PositionSwitchedCalibration.get_temperature_correction_factor(
                    paired_hdu, zenith_opacity
                )
            )
            if correction_factor is None:
                continue

            baseline_subtracted_spectrum.intensity *= correction_factor
            baseline_subtracted_spectrum.noise *= correction_factor

        lidx, ridx = numpy.searchsorted(
            frequency, baseline_subtracted_spectrum.frequency[[0, -1]]
        )

        interpolated_intensity = numpy.interp(
            frequency[lidx:ridx],
            baseline_subtracted_spectrum.frequency,
            baseline_subtracted_spectrum.intensity,
            left=0,
            right=0,
        )
        interpolated_noise = numpy.interp(
            frequency[lidx:ridx],
            baseline_subtracted_spectrum.frequency,
            baseline_subtracted_spectrum.noise,
            left=0,
            right=0,
        )

        weighted_intensity[lidx:ridx] += interpolated_noise**-2 * interpolated_intensity
        inverse_variance[lidx:ridx] += interpolated_noise**-2

    output_directory: pathlib.Path = (
        pathlib.Path.cwd() if args.output_directory is None else args.output_directory
    )
    output_path = output_directory / sdfits.path.name
    numpy.savez_compressed(output_path, frequency=frequency, weighted_intensity=weighted_intensity, inverse_variance=inverse_variance)
