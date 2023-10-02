import collections
import enum
import functools
import typing

import astropy.timeseries
import bottleneck
import loguru
import numpy
import numpy.polynomial.polynomial as poly
import numpy.typing
import scipy.special

Baseline = typing.Callable[
    [numpy.typing.NDArray[numpy.floating]], numpy.typing.NDArray[numpy.floating]
]
BaselineSupplementaryInfo = dict[str, typing.Any]


def _fit_polynomial_baseline(
    frequency: numpy.typing.NDArray[numpy.floating],
    intensity: numpy.typing.NDArray[numpy.floating],
    noise: numpy.typing.NDArray[numpy.floating] | None,
    degree: int,
) -> tuple[Baseline, BaselineSupplementaryInfo]:
    if noise is None:
        polynomial = poly.Polynomial.fit(frequency, intensity, degree)
    else:
        polynomial = poly.Polynomial.fit(frequency, intensity, degree, w=1 / noise)
    return polynomial, {"polynomial": polynomial}


def _auto_fit_polynomial_baseline(
    frequency: numpy.typing.NDArray[numpy.floating],
    intensity: numpy.typing.NDArray[numpy.floating],
    noise: numpy.typing.NDArray[numpy.floating] | None,
    residual_threshold: float | None,
    residual_half_moving_window: int,
    polynomial_options: dict[str, typing.Any],
) -> tuple[Baseline, BaselineSupplementaryInfo] | None:
    degree = polynomial_options.pop("degree", None)
    min_degree = polynomial_options.pop("min_degree", 0)
    max_degree = polynomial_options.pop("max_degree", 1024)

    if degree is None:
        if noise is None:
            loguru.logger.warning(
                "Finding minimum degree using residual requires noise but this spectrum does not have noise."
            )
            return None

        if residual_threshold is None:
            loguru.logger.warning(
                "Finding minimum degree using residual requires specifying residual_threshold."
            )
            return None

        if min_degree > max_degree:
            loguru.logger.warning(f"Require {min_degree = } <= {max_degree = }.")
            return None

        # Find minimum degree such that residual < residual_threshold using binary search.
        # If no such degree is found, use max_degree.
        success = False
        cached_result: tuple[Baseline, BaselineSupplementaryInfo]
        lb, ub = min_degree, max_degree + 1
        while lb < ub:
            degree = lb + (ub - lb) // 2
            baseline, baseline_info = _fit_polynomial_baseline(
                frequency, intensity, noise, degree, **polynomial_options
            )
            baseline_info["residual"] = residual = _compute_residual(
                intensity - baseline(frequency),
                noise,
                residual_half_moving_window,
            )
            if residual < residual_threshold:
                baseline_info["success"] = success = True
                cached_result = baseline, baseline_info
                ub = degree
            else:
                if not success:
                    baseline_info["success"] = success = False
                    cached_result = baseline, baseline_info
                lb = degree + 1
        if not success:
            degree = max_degree
            loguru.logger.warning(
                f"Cannot find a degree <= {max_degree = } such that residual < {residual_threshold = }. Using {degree = }."
            )
        baseline, baseline_info = cached_result
        intensity -= baseline(frequency)
        return baseline, baseline_info
    else:
        baseline, baseline_info = _fit_polynomial_baseline(
            frequency, intensity, noise, degree, **polynomial_options
        )
        intensity -= baseline(frequency)

        if residual_threshold is not None:
            if noise is None:
                loguru.logger.warning(
                    "Computing residual requires noise but this spectrum does not have noise."
                )
                return None

            baseline_info["residual"] = residual = _compute_residual(
                intensity,
                noise,
                residual_half_moving_window,
            )
            if residual < residual_threshold:
                baseline_info["success"] = True
            else:
                baseline_info["success"] = False
                loguru.logger.warning(f"{residual = } >= {residual_threshold = }.")

        return baseline, baseline_info


def _fit_lomb_scargle_baseline(
    frequency: numpy.typing.NDArray[numpy.floating],
    intensity: numpy.typing.NDArray[numpy.floating],
    noise: numpy.typing.NDArray[numpy.floating] | None,
    max_cycle: float | None = None,
) -> tuple[Baseline, BaselineSupplementaryInfo]:
    if noise is None:
        lomb_scargle = astropy.timeseries.LombScargle(frequency, intensity)
    else:
        lomb_scargle = astropy.timeseries.LombScargle(frequency, intensity, noise)

    if max_cycle is None:
        ls_frequency, power = lomb_scargle.autopower(method="fast")
    else:
        min_frequency = (frequency.max() - frequency.min()) / max_cycle
        max_ls_frequency = 1 / min_frequency
        ls_frequency, power = lomb_scargle.autopower(
            method="fast", maximum_frequency=max_ls_frequency
        )
    best_ls_frequency = ls_frequency[numpy.argmax(power)]
    return functools.partial(lomb_scargle.model, frequency=best_ls_frequency), {
        "lomb_scargle": lomb_scargle,
        "best_ls_frequency": best_ls_frequency,
    }


def _auto_fit_lomb_scargle_baseline(
    frequency: numpy.typing.NDArray[numpy.floating],
    intensity: numpy.typing.NDArray[numpy.floating],
    noise: numpy.typing.NDArray[numpy.floating] | None,
    residual_threshold: float | None,
    residual_half_moving_window: int,
    lomb_scargle_options: dict[str, typing.Any],
) -> tuple[list[Baseline], list[BaselineSupplementaryInfo]] | None:
    num_terms = lomb_scargle_options.pop("num_terms", None)
    min_num_terms = lomb_scargle_options.pop("min_num_terms", 0)
    max_num_terms = lomb_scargle_options.pop("max_num_terms", 128)

    baseline_list: list[Baseline] = []
    baseline_info_list: list[BaselineSupplementaryInfo] = []

    if num_terms is None:
        if noise is None:
            loguru.logger.warning(
                "Finding number of terms using residual requires noise but this spectrum does not have noise."
            )
            return None

        if residual_threshold is None:
            loguru.logger.warning(
                "Finding minimum number of terms using residual which requires specifying residual_threshold."
            )
            return None

        if min_num_terms > max_num_terms:
            loguru.logger.warning(f"Require {min_num_terms = } <= {max_num_terms = }.")
            return None

        if min_num_terms == 0:
            residual = _compute_residual(intensity, noise, residual_half_moving_window)
            if residual < residual_threshold:
                return [], []

        # Find minimum number of terms such that residual < residual_threshold using linear search.
        # If no such degree is found, use max_num_terms.
        success = False
        for num_terms in range(1, max_num_terms + 1):
            baseline, baseline_info = _fit_lomb_scargle_baseline(
                frequency, intensity, noise, **lomb_scargle_options
            )
            intensity = intensity - baseline(frequency)

            baseline_info["residual"] = residual = _compute_residual(
                intensity, noise, residual_half_moving_window
            )
            if residual < residual_threshold:
                baseline_info["success"] = success = True
            else:
                baseline_info["success"] = success = False

            baseline_list.append(baseline)
            baseline_info_list.append(baseline_info)

            if success and num_terms >= min_num_terms:
                break

        if not success:
            loguru.logger.warning(
                f"Cannot find a number of terms <= {max_num_terms = } such that residual < {residual_threshold = }. Using {num_terms = }."
            )
        return baseline_list, baseline_info_list
    else:
        for _ in range(num_terms):
            baseline, baseline_info = _fit_lomb_scargle_baseline(
                frequency, intensity, noise, **lomb_scargle_options
            )
            intensity -= baseline(frequency)

            if residual_threshold is not None:
                if noise is None:
                    loguru.logger.warning(
                        "Computing residual requires noise but this spectrum does not have noise."
                    )
                    return None

                baseline_info["residual"] = residual = _compute_residual(
                    intensity,
                    noise,
                    residual_half_moving_window,
                )
                if residual < residual_threshold:
                    baseline_info["success"] = True
                else:
                    baseline_info["success"] = False

            baseline_list.append(baseline)
            baseline_info_list.append(baseline_info)

        if not baseline_info_list[-1]["success"]:
            residual = baseline_info_list[-1]["residual"]
            loguru.logger.warning(f"{residual = } >= {residual_threshold = }.")
        return baseline_list, baseline_info_list


def _centered_move_sum(
    arr: numpy.typing.NDArray[typing.Any], half_moving_window: int
) -> numpy.typing.NDArray[numpy.floating]:
    moving_window = 2 * half_moving_window + 1
    move_sum = bottleneck.move_sum(arr, window=moving_window, min_count=1)
    move_sum[:-half_moving_window] = move_sum[half_moving_window:]
    move_sum[-half_moving_window:] = bottleneck.move_sum(
        arr[-moving_window:][::-1], window=moving_window, min_count=1
    )[half_moving_window : 2 * half_moving_window][::-1]
    return move_sum


def _centered_move_mean(
    arr: numpy.typing.NDArray[typing.Any], half_moving_window: int
) -> numpy.typing.NDArray[numpy.floating]:
    moving_window = 2 * half_moving_window + 1
    move_mean = bottleneck.move_mean(arr, window=moving_window, min_count=1)
    move_mean[:-half_moving_window] = move_mean[half_moving_window:]
    move_mean[-half_moving_window:] = bottleneck.move_mean(
        arr[-moving_window:][::-1], window=moving_window, min_count=1
    )[half_moving_window : 2 * half_moving_window][::-1]
    return move_mean


def _compute_residual(
    intensity: numpy.typing.NDArray[numpy.floating],
    noise: numpy.typing.NDArray[numpy.floating],
    half_moving_window: int,
):
    return numpy.sqrt(numpy.nanmean(
        _centered_move_mean(
            intensity / noise,
            half_moving_window,
        )
        ** 2
    ))


class Spectrum:
    frequency: numpy.typing.NDArray[numpy.floating]
    intensity: numpy.typing.NDArray[numpy.floating]
    noise: numpy.typing.NDArray[numpy.floating] | None
    flag: numpy.typing.NDArray[numpy.object_] | None

    class FlagReason(enum.Enum):
        NOT_FLAGGED = 0
        CHUNK_EDGES = 1
        RFI = 2

    def __init__(
        self,
        frequency: numpy.typing.ArrayLike,
        intensity: numpy.typing.ArrayLike,
        noise: numpy.typing.ArrayLike | None,
        no_flag: bool = False,
    ):
        self.frequency = numpy.array(frequency)
        self.intensity = numpy.array(intensity)
        if not (self.frequency.shape == self.intensity.shape):
            loguru.logger.error("Frequency and intensity do not have the same shape.")
            self.frequency = self.intensity = numpy.empty(0)
            return

        if noise is None:
            self.noise = None
        else:
            self.noise = numpy.array(noise)
            if not (self.frequency.shape == self.noise.shape):
                loguru.logger.error(
                    "Frequency, intensity and noise do not have the same shape."
                )
                self.frequency = self.intensity = self.noise = numpy.empty(0)
                return

        if no_flag:
            self.flag = None
        else:
            self.flag = numpy.full_like(
                self.frequency, Spectrum.FlagReason.NOT_FLAGGED, dtype=object
            )

    @property
    def flagged(self) -> numpy.typing.NDArray[numpy.bool_]:
        if self.flag is None:
            loguru.logger.warning("This spectrum have no flags.")
            return numpy.full_like(self.frequency, False)

        return self.flag != Spectrum.FlagReason.NOT_FLAGGED

    def detect_signal(
        self, nadjacent, confidence
    ) -> numpy.typing.NDArray[numpy.bool_] | None:
        if self.noise is None:
            loguru.logger.warning("This spectrum has no noise.")
            return None

        # TODO: This implementation can be optimized using bottleneck. It
        # requires a expanding the parenthesis manually and rewrite p, q, r, u,
        # v and w. It may hurt readability so performance should be measured
        # before actually optimizing it.
        #
        # Here, we want to calculate for each point the minimum chi-square that
        # (2n + 1) adjacent points centered on the given point lies on a
        # straight line mx + c.
        # We define chi-squared = Sum_i (((mx_i + c) - y_i) / s_i)^2 and
        # minimizing it w.r.t. m and c gives a system of equation of the form:
        #   p * m + q * c = r
        #   u * m + v * c = w
        # where
        width = 2 * nadjacent + 1
        x = (
            numpy.lib.stride_tricks.sliding_window_view(self.frequency, width)
            - self.frequency[nadjacent:-nadjacent, None]
        )
        y = (
            numpy.lib.stride_tricks.sliding_window_view(self.intensity, width)
            - self.intensity[nadjacent:-nadjacent, None]
        )
        s = numpy.lib.stride_tricks.sliding_window_view(self.noise, width)
        p = (x**2 / s**2).sum(axis=-1)
        q = (x / s**2).sum(axis=-1)
        r = (x * y / s**2).sum(axis=-1)
        u = q
        v = (1 / s**2).sum(axis=-1)
        w = (y / s**2).sum(axis=-1)
        # Solve the system of equation using Cramer's rule gives:
        delta = p * v - u * q
        m = (r * v - w * q) / delta
        c = (p * w - u * r) / delta
        # We can get chi-squared by definition
        chi_squared = (((m[:, None] * x + c[:, None] - y) / s) ** 2).sum(axis=-1)
        is_signal = numpy.where(scipy.special.chdtr(width, chi_squared) > confidence)[0]

        res = numpy.zeros_like(self.frequency, dtype=bool)
        for i in range(width):
            res[i : delta.size + i][is_signal] = True
        return res

    def flag_rfi(self, nadjacent: int = 3, confidence: float = 0.999999):
        if self.flag is None:
            loguru.logger.warning("This spectrum have no flags.")
            return

        is_rfi = self.detect_signal(nadjacent, confidence)
        self.flag[is_rfi] = Spectrum.FlagReason.RFI

    def flag_head_tail(
        self, *, fraction: float | None = None, nchannel: int | None = None
    ):
        if self.flag is None:
            loguru.logger.warning("This spectrum have no flags.")
            return
        if nchannel is None:
            if fraction is None:
                loguru.logger.error(
                    "Either but not both fraction or nchannel should be set."
                )
                return
            nchannel = int(self.frequency.size * fraction)

        self.flag[:nchannel] = self.flag[-nchannel:] = Spectrum.FlagReason.CHUNK_EDGES

    def fit_baseline(
        self,
        method: typing.Literal["hybrid", "polynomial", "lomb-scargle"] = "hybrid",
        mask: numpy.typing.NDArray[numpy.bool_] | None = None,
        residual_threshold: float | None = None,
        residual_half_moving_window: int = 512,
        polynomial_options: dict[str, typing.Any] = {},
        lomb_scargle_options: dict[str, typing.Any] = {},
    ) -> tuple[Baseline, BaselineSupplementaryInfo] | None:
        fitted = ~self.flagged if mask is None else ~(mask | self.flagged)
        frequency = self.frequency[fitted]
        intensity = self.intensity[fitted]
        noise = None if self.noise is None else self.noise[fitted]

        if residual_threshold is not None and noise is None:
            loguru.logger.warning(
                "Computing residual requires noise but this spectrum does not have noise."
            )
            return None

        polynomial_options = polynomial_options.copy()
        lomb_scargle_options = lomb_scargle_options.copy()

        all_baseline: list[Baseline] = []
        all_baseline_info: list[BaselineSupplementaryInfo] = []

        if method == "hybrid" or method == "polynomial":
            polynomial_result = _auto_fit_polynomial_baseline(
                frequency,
                intensity,
                noise,
                residual_threshold,
                residual_half_moving_window,
                polynomial_options,
            )
            if polynomial_result is None:
                return None

            baseline, baseline_info = polynomial_result
            all_baseline.append(baseline)
            all_baseline_info.append(baseline_info)

        if method == "hybrid" or method == "lomb-scargle":
            lomb_scargle_result = _auto_fit_lomb_scargle_baseline(
                frequency,
                intensity,
                noise,
                residual_threshold,
                residual_half_moving_window,
                lomb_scargle_options,
            )
            if lomb_scargle_result is None:
                return None

            baseline_list, baseline_info_list = lomb_scargle_result
            all_baseline.extend(baseline_list)
            all_baseline_info.extend(baseline_info_list)

        def combined_baseline(
            frequency: numpy.typing.NDArray[numpy.floating],
        ) -> numpy.typing.NDArray[numpy.floating]:
            if len(all_baseline) == 0:
                return numpy.zeros_like(frequency)
            if len(all_baseline) == 1:
                return all_baseline[0](frequency)
            return numpy.sum([baseline(frequency) for baseline in all_baseline], axis=0)

        combined_baseline_info = collections.defaultdict(list)
        for baseline_info in all_baseline_info:
            for key, val in baseline_info.items():
                combined_baseline_info[key].append(val)
        return combined_baseline, combined_baseline_info

    def get_moving_averaged_spectrum(
        self,
        half_moving_window: int = 512,
        mask: numpy.typing.NDArray[numpy.bool_] | None = None,
    ) -> "Spectrum":
        not_flagged = ~self.flagged if mask is None else ~(mask | self.flagged)
        count = _centered_move_sum(not_flagged, half_moving_window)
        intensity = (
            _centered_move_sum(
                numpy.where(not_flagged, self.intensity, numpy.nan), half_moving_window
            )
            / count
        )
        noise = (
            None
            if self.noise is None
            else numpy.sqrt(
                _centered_move_sum(
                    numpy.where(not_flagged, self.noise**2, numpy.nan),
                    half_moving_window,
                )
            )
            / count
        )
        return Spectrum(self.frequency, intensity, noise, no_flag=True)
