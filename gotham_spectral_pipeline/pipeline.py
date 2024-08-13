import argparse
import collections
import loguru
import math
import sys
import traceback
from typing_extensions import Self

import tqdm  # type: ignore

from . import SigRefPairedRows, SigRefPairedHDUList
from . import GbtTsysLookupTable, GbtTsysHybridSelector, TsysThresholdSelector
from . import BeamEfficiency, PositionSwitchedCalibration, SDFits, ZenithOpacity
from . import Exposure, ExposureAggregator
from . import Spectrum, SpectrumAggregator

__all__ = [
    "Pipeline",
]


class Pipeline:

    class Input:
        sdfits: SDFits
        zenith_opacity: ZenithOpacity | None
        beam_efficiency: BeamEfficiency | None
        paired_rows: list[SigRefPairedRows]

    class FilteredIntegration:
        paired_row: SigRefPairedRows
        exposure: Exposure
        spectrum: Spectrum
        correction_factor: Spectrum | None

    class PreBaselineOutput:
        filtered_integrations: list["Pipeline.FilteredIntegration"]

    class BaselineOutput:
        filtered_integrations: list["Pipeline.FilteredIntegration"]

    class PostBaselineOutput:
        pass

    class Output:
        success: bool = False
        reason: str = "Unknown reason"
        integration_dropped_reason: collections.Counter[str]

        total_exposure: Exposure
        exposure: Exposure
        spectrum: Spectrum

        def __init__(self):
            self.integration_dropped_reason = collections.Counter()

    class Options:
        channel_width: float
        flag_head_tail_channel_number: int
        max_rfi_channel: int
        Tsys_dynamic_threshold: bool
        Tsys_dynamic_threshold_selector: str
        Tsys_min_threshold: float
        Tsys_max_threshold: float
        Tsys_min_success_rate: float

        @classmethod
        def from_dict(cls, d: dict) -> Self:
            res = cls()
            for field_name, field_type in cls.__annotations__.items():
                if field_name not in d:
                    continue
                if not isinstance(d[field_name], field_type):
                    loguru.logger.error(
                        f"Field {field_name} has type {type(d[field_name])}. Expected {field_type}"
                    )
                    continue
                setattr(res, field_name, d[field_name])
            return res

        @classmethod
        def from_namespace(cls, ns: argparse.Namespace) -> Self:
            return cls.from_dict(vars(ns))

    class Halt(RuntimeError):
        pass

    _input: Input
    _pre_baseline_output: PreBaselineOutput
    _baseline_output: BaselineOutput
    _post_baseline_output: PostBaselineOutput
    _output: Output
    _options: Options

    def __init__(
        self,
        sdfits: SDFits,
        zenith_opacity: ZenithOpacity | None,
        beam_efficiency: BeamEfficiency | None,
        paired_rows: list[SigRefPairedRows],
        options: Options,
    ):
        self.sdfits = sdfits
        self.zenith_opacity = zenith_opacity
        self.beam_efficiency = beam_efficiency
        self.paired_rows = paired_rows

        self._input = self.Input()
        self._pre_baseline_output = self.PreBaselineOutput()
        self._baseline_output = self.BaselineOutput()
        self._post_baseline_output = self.PostBaselineOutput()
        self._output = self.Output()

        self._input.sdfits = sdfits
        self._input.zenith_opacity = zenith_opacity
        self._input.beam_efficiency = beam_efficiency
        self._input.paired_rows = paired_rows
        self._options = options

    def _get_tsys_threshold_selector(self) -> TsysThresholdSelector | None:
        if self._options.Tsys_dynamic_threshold:
            if self._options.Tsys_dynamic_threshold_selector == "LookupTable":
                return GbtTsysLookupTable()
            elif self._options.Tsys_dynamic_threshold_selector == "Hybrid":
                return GbtTsysHybridSelector()
            else:
                loguru.logger.error(
                    f"Unexpected Tsys_dynamic_threshold_selector = {self._options.Tsys_dynamic_threshold_selector}."
                )
                return None
        return None

    def _check_tsys_threshold(
        self,
        spectrum_metadata: dict,
        tsys_threshold_selector: TsysThresholdSelector | None,
        tsys_stats: dict[str, int],
    ) -> bool:
        Tsys_min_threshold = self._options.Tsys_min_threshold
        Tsys_max_threshold = self._options.Tsys_max_threshold
        if tsys_threshold_selector is not None:
            dynamic_threshold = tsys_threshold_selector.get_threshold(
                spectrum_metadata["ObsFreq"]
            )
            if dynamic_threshold is None:
                loguru.logger.error("Tsys threshold selector returned None.")
                return False
            Tsys_min_threshold *= dynamic_threshold
            Tsys_max_threshold *= dynamic_threshold

        tsys_stats["total"] += 1
        if not spectrum_metadata["Tsys"] > Tsys_min_threshold:
            self._output.integration_dropped_reason["Tsys exceeds min threshold"] += 1
            return False
        if not spectrum_metadata["Tsys"] < Tsys_max_threshold:
            self._output.integration_dropped_reason["Tsys exceeds max threshold"] += 1
            return False
        tsys_stats["succeed"] += 1
        return True

    def _check_tsys_stats(self, tsys_stats: dict[str, int]) -> bool:
        if tsys_stats["total"] == 0:
            tsys_success_rate = 0.0
        else:
            tsys_success_rate = tsys_stats["succeed"] / tsys_stats["total"]
        if tsys_success_rate < self._options.Tsys_min_success_rate:
            return False
        return True

    def _check_num_rfi_channel(
        self,
        spectrum: Spectrum,
    ) -> bool:
        assert spectrum.flag is not None
        if self._options.max_rfi_channel >= 0:
            rfi_in_body_count = (
                ~spectrum.flagged(Spectrum.FlagReason.CHUNK_EDGES)
                & spectrum.flagged(
                    Spectrum.FlagReason.FREQUENCY_DOMAIN_RFI
                    | Spectrum.FlagReason.TIME_DOMAIN_RFI
                )
            ).sum()
            if rfi_in_body_count > self._options.max_rfi_channel:
                self._output.integration_dropped_reason["Too many RFI channels"] += 1
                return False
        return True

    def _get_correction_factor(
        self, sigrefpair: SigRefPairedHDUList
    ) -> Spectrum | None:
        correction_factors = []
        if self._input.zenith_opacity is not None:
            opacity_correction_factor = (
                PositionSwitchedCalibration.get_opacity_correction_factor(
                    sigrefpair["sig"]["caloff"], self._input.zenith_opacity
                )
            )
            if opacity_correction_factor is None:
                raise self.Halt("Opacity temperature correction enabled but failed.")
            correction_factors.append(opacity_correction_factor)
        if self._input.beam_efficiency is not None:
            efficiency_correction_factor = (
                PositionSwitchedCalibration.get_efficiency_correction_factor(
                    sigrefpair["sig"]["caloff"], self._input.beam_efficiency
                )
            )
            if efficiency_correction_factor is None:
                raise self.Halt("Beam efficiency correction enabled but failed.")
            correction_factors.append(efficiency_correction_factor)
        if len(correction_factors) == 0:
            return None
        return math.prod(correction_factors)

    def _get_debug_indices(self, paired_row: SigRefPairedRows):
        return {
            f"{sigref},{calonoff}": int(paired_row[sigref][calonoff]["INDEX"].iloc[0])
            for sigref in paired_row
            for calonoff in paired_row[sigref]
        }

    def _run_stage_pre_baseline(self) -> bool:
        total_exposure_aggregator = ExposureAggregator(
            ExposureAggregator.LinearTransformer(self._options.channel_width)
        )

        tsys_threshold_selector = self._get_tsys_threshold_selector()
        tsys_stats = dict(succeed=0, total=0)

        self._pre_baseline_output.filtered_integrations = list()
        for paired_row in tqdm.tqdm(
            self._input.paired_rows, dynamic_ncols=True, smoothing=0.0, leave=False
        ):
            try:
                sigrefpair = paired_row.get_paired_hdu(self._input.sdfits)
                exposure = PositionSwitchedCalibration.get_exposure(sigrefpair["sig"])
                if exposure is None:
                    self._output.integration_dropped_reason["No exposure returned"] += 1
                    continue
                total_exposure_aggregator.merge(exposure)

                if PositionSwitchedCalibration.should_be_discarded(sigrefpair):
                    self._output.integration_dropped_reason["Failed prechecks"] += 1
                    continue

                (
                    spectrum,
                    spectrum_metadata,
                ) = PositionSwitchedCalibration.get_calibrated_spectrum(
                    sigrefpair, freq_kwargs=dict(unit="Hz"), return_metadata=True
                )
                if spectrum is None:
                    self._output.integration_dropped_reason[
                        "No calibrated spectrum returned"
                    ] += 1
                    continue

                if not self._check_tsys_threshold(
                    spectrum_metadata,
                    tsys_threshold_selector,
                    tsys_stats,
                ):
                    continue

                (
                    spectrum.flag_nan()
                    .flag_head_tail(
                        nchannel=self._options.flag_head_tail_channel_number
                    )
                    .flag_time_domain_rfi(spectrum_metadata)
                    .flag_frequency_domain_rfi()
                    .flag_valid_data()
                )

                if not self._check_num_rfi_channel(spectrum):
                    continue

                exposure.exposure[
                    ~spectrum.flagged(Spectrum.FlagReason.VALID_DATA)
                ] = 0.0

                filtered_integration = self.FilteredIntegration()
                filtered_integration.paired_row = paired_row
                filtered_integration.exposure = exposure
                filtered_integration.spectrum = spectrum
                filtered_integration.correction_factor = self._get_correction_factor(
                    sigrefpair
                )
                self._pre_baseline_output.filtered_integrations.append(
                    filtered_integration
                )
            except self.Halt as e:
                loguru.logger.critical(*e.args)
                tqdm.tqdm.write(*e.args)
                sys.exit(1)
            except Exception:
                loguru.logger.critical(
                    f"Uncaught exception while working on {self._get_debug_indices(paired_row)}\n{traceback.format_exc()}"
                )
                self._output.integration_dropped_reason["Uncaught exception"] += 1

        self._output.total_exposure = total_exposure_aggregator.get_spectrum()

        if not self._check_tsys_stats(tsys_stats):
            self._output.reason = "Tsys threshold success rate less than required."
            return False

        return True

    def _run_stage_baseline(self) -> bool:
        pre_baseline_aggregated_spectrum = (
            SpectrumAggregator(
                SpectrumAggregator.LinearTransformer(self._options.channel_width)
            )
            .merge_all(
                spectrum - spectrum.from_callable(baseline_result[0])
                for spectrum, baseline_result in (
                    (
                        filtered_integration.spectrum,
                        filtered_integration.spectrum.fit_baseline(
                            method="polynomial", polynomial_options=dict(degree=20)
                        ),
                    )
                    for filtered_integration in self._pre_baseline_output.filtered_integrations
                )
                if baseline_result is not None
            )
            .get_spectrum()
            # Flag strong signals where baseline fitting is affected a lot
            .flag_signal(
                nadjacent=31,
                ignore_flags=Spectrum.FlagReason.CHUNK_EDGES
                | Spectrum.FlagReason.FREQUENCY_DOMAIN_RFI
                | Spectrum.FlagReason.TIME_DOMAIN_RFI,
            )
            # Flag weak signals that require more accurate baseline fitting
            .flag_signal(
                nadjacent=dict(baseline=255, chisq=31),
                ignore_flags=Spectrum.FlagReason.CHUNK_EDGES
                | Spectrum.FlagReason.FREQUENCY_DOMAIN_RFI
                | Spectrum.FlagReason.TIME_DOMAIN_RFI
                | Spectrum.FlagReason.SIGNAL,
            )
        )

        self._baseline_output.filtered_integrations = list()
        for filtered_integration in tqdm.tqdm(
            self._pre_baseline_output.filtered_integrations,
            dynamic_ncols=True,
            smoothing=0.0,
            leave=False,
        ):
            paired_row = filtered_integration.paired_row
            exposure = filtered_integration.exposure
            spectrum = filtered_integration.spectrum
            correction_factor = filtered_integration.correction_factor
            try:
                spectrum.copy_flags(
                    Spectrum.FlagReason.SIGNAL, pre_baseline_aggregated_spectrum
                )

                baseline_result = spectrum.fit_baseline(
                    method="hybrid",
                    polynomial_options=dict(max_degree=20),
                    lomb_scargle_options=dict(
                        min_num_terms=0, max_num_terms=40, max_cycle=32
                    ),
                    residual_threshold=0.1,
                )
                if baseline_result is None:
                    self._output.integration_dropped_reason[
                        "Failed baseline fitting"
                    ] += 1
                    continue

                baseline, _ = baseline_result
                baseline_substracted_spectrum = spectrum - spectrum.from_callable(
                    baseline
                )
                if correction_factor is not None:
                    baseline_substracted_spectrum *= correction_factor

                filtered_integration = self.FilteredIntegration()
                filtered_integration.paired_row = paired_row
                filtered_integration.exposure = exposure
                filtered_integration.spectrum = baseline_substracted_spectrum
                self._baseline_output.filtered_integrations.append(filtered_integration)
            except self.Halt as e:
                loguru.logger.critical(*e.args)
                tqdm.tqdm.write(*e.args)
                sys.exit(1)
            except Exception:
                loguru.logger.critical(
                    f"Uncaught exception while working on {self._get_debug_indices(paired_row)}\n{traceback.format_exc()}"
                )
                self._output.integration_dropped_reason["Uncaught exception"] += 1
        return True

    def _run_stage_post_baseline(self) -> bool:
        self._output.exposure = (
            ExposureAggregator(
                ExposureAggregator.LinearTransformer(self._options.channel_width)
            )
            .merge_all(self._output.total_exposure.split(), init_only=True)
            .merge_all(
                filtered_integration.exposure
                for filtered_integration in self._baseline_output.filtered_integrations
            )
            .get_spectrum()
        )
        self._output.spectrum = (
            SpectrumAggregator(
                SpectrumAggregator.LinearTransformer(self._options.channel_width)
            )
            .merge_all(
                filtered_integration.spectrum
                for filtered_integration in self._baseline_output.filtered_integrations
            )
            .get_spectrum()
        )

        import numpy

        if numpy.any(numpy.isnan(self._output.total_exposure.frequency)):
            print(f"{numpy.isnan(self._output.total_exposure.frequency).sum() = }")
        if numpy.any(numpy.isnan(self._output.total_exposure.exposure)):
            print(f"{numpy.isnan(self._output.total_exposure.exposure).sum() = }")
        if numpy.any(numpy.isnan(self._output.exposure.frequency)):
            print(f"{numpy.isnan(self._output.exposure.frequency).sum() = }")
        if numpy.any(numpy.isnan(self._output.exposure.exposure)):
            print(f"{numpy.isnan(self._output.exposure.exposure).sum() = }")
        if numpy.any(numpy.isnan(self._output.spectrum.frequency)):
            print(f"{numpy.isnan(self._output.spectrum.frequency).sum() = }")
        if numpy.any(numpy.isnan(self._output.spectrum.intensity)):
            print(f"{numpy.isnan(self._output.spectrum.intensity).sum() = }")

        return True

    def calibrate(self) -> Self:
        if not self._run_stage_pre_baseline():
            return self
        if not self._run_stage_baseline():
            return self
        if not self._run_stage_post_baseline():
            return self
        self._output.success = True
        self._output.reason = "Success"
        return self

    def get_output(self) -> Output:
        return self._output
