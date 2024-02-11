import collections
import enum
import functools
import itertools
import typing

import astropy.timeseries  # type: ignore
import bottleneck  # type: ignore
import loguru
import numpy
import numpy.polynomial.polynomial as poly
import numpy.typing
import scipy.special  # type: ignore
from sortedcontainers import SortedList  # type: ignore

__all__ = ["Spectrum", "SpectrumAggregator"]

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
    return numpy.sqrt(
        numpy.nanmean(
            _centered_move_mean(
                intensity / noise,
                half_moving_window,
            )
            ** 2
        )
    )


class Spectrum:
    frequency: numpy.typing.NDArray[numpy.floating] | None
    intensity: numpy.typing.NDArray[numpy.floating]
    noise: numpy.typing.NDArray[numpy.floating] | None
    flag: numpy.typing.NDArray[numpy.integer] | None

    class FlagReason(enum.Enum):
        NOT_FLAGGED = 0
        CHUNK_EDGES = 1 << 0
        RFI = 1 << 1

    def __init__(
        self,
        *,
        intensity: numpy.typing.ArrayLike,
        frequency: numpy.typing.ArrayLike | None = None,
        noise: numpy.typing.ArrayLike | None = None,
        flag: numpy.typing.ArrayLike | None = None,
    ):
        self.intensity = numpy.array(intensity, dtype=numpy.float64)

        if frequency is None:
            self.frequency = None
        else:
            self.frequency = numpy.array(frequency, dtype=numpy.float64)

        if noise is None:
            self.noise = None
        else:
            self.noise = numpy.array(noise, dtype=numpy.float64)

        if flag is None:
            self.flag = None
        else:
            self.flag = numpy.array(flag, dtype=numpy.int32)

        shapes = {
            ndarray.shape
            for ndarray in [self.frequency, self.intensity, self.noise, self.flag]
            if ndarray is not None
        }
        if len(shapes) != 1:
            loguru.logger.error("Shapes of input arrays are not identical.")
            return

        if self.frequency is not None:
            self._sort_by_frequency()

    def _sort_by_frequency(self):
        is_sorted = lambda a: numpy.all(a[1:] >= a[:-1])

        if is_sorted(self.frequency):
            return

        if is_sorted(self.frequency[::-1]):
            self.intensity = self.intensity[::-1]
            self.frequency = self.frequency[::-1]
            if self.noise is not None:
                self.noise = self.noise[::-1]
            if self.flag is not None:
                self.flag = self.flag[::-1]
            return

        sorted_idx = numpy.argsort(self.frequency)
        self.intensity = self.intensity[sorted_idx]
        self.frequency = self.frequency[sorted_idx]
        if self.noise is not None:
            self.noise = self.noise[sorted_idx]
        if self.flag is not None:
            self.flag = self.flag[sorted_idx]

    def __add__(self, other: "Spectrum") -> "Spectrum":
        if isinstance(other, Spectrum):
            intensity = self.intensity + other.intensity

            if self.frequency is None or other.frequency is None:
                frequency = None
            else:
                # Assume both spectra have the same frequency and use the one of the left operand
                frequency = self.frequency

            if self.noise is None or other.noise is None:
                noise = None
            else:
                noise = numpy.sqrt(numpy.square(self.noise) + numpy.square(other.noise))

            if self.flag is None or other.flag is None:
                flag = None
            else:
                flag = self.flag | other.flag

            return Spectrum(
                intensity=intensity, frequency=frequency, noise=noise, flag=flag
            )

        return NotImplemented

    def __sub__(self, other: "Spectrum") -> "Spectrum":
        if isinstance(other, Spectrum):
            intensity = self.intensity - other.intensity

            if self.frequency is None or other.frequency is None:
                frequency = None
            else:
                # Assume both spectra have the same frequency and use the one of the left operand
                frequency = self.frequency

            if self.noise is None or other.noise is None:
                noise = None
            else:
                noise = numpy.sqrt(numpy.square(self.noise) + numpy.square(other.noise))

            if self.flag is None or other.flag is None:
                flag = None
            else:
                flag = self.flag | other.flag

            return Spectrum(
                intensity=intensity, frequency=frequency, noise=noise, flag=flag
            )

        return NotImplemented

    def __mul__(self, other: "int | float | Spectrum") -> "Spectrum":
        if isinstance(other, int) or isinstance(other, float):
            if self.frequency is None:
                frequency = None
            else:
                frequency = self.frequency

            intensity = self.intensity * other

            if self.noise is None:
                noise = None
            else:
                noise = self.noise * other

            if self.flag is None:
                flag = None
            else:
                flag = self.flag

            return Spectrum(
                intensity=intensity, frequency=frequency, noise=noise, flag=flag
            )

        if isinstance(other, Spectrum):
            if self.frequency is None or other.frequency is None:
                frequency = None
            else:
                # Assume both spectra have the same frequency and use the one of the left operand
                frequency = self.frequency

            intensity = self.intensity * other.intensity

            if self.noise is None or other.noise is None:
                noise = None
            else:
                noise = intensity * numpy.sqrt(
                    numpy.square(self.noise / self.intensity)
                    + numpy.square(other.noise / other.intensity)
                )

            if self.flag is None or other.flag is None:
                flag = None
            else:
                flag = self.flag | other.flag

            return Spectrum(
                intensity=intensity, frequency=frequency, noise=noise, flag=flag
            )

        return NotImplemented

    def __rmul__(self, other: int | float):
        return self * other

    def __truediv__(self, other: "int | float | Spectrum") -> "Spectrum":
        if isinstance(other, int) or isinstance(other, float):
            if self.frequency is None:
                frequency = None
            else:
                frequency = self.frequency

            intensity = self.intensity / other

            if self.noise is not None:
                noise = self.noise / other
            else:
                noise = None

            if self.flag is None:
                flag = None
            else:
                flag = self.flag

            return Spectrum(
                intensity=intensity, frequency=frequency, noise=noise, flag=flag
            )

        if isinstance(other, Spectrum):
            if self.frequency is None or other.frequency is None:
                frequency = None
            else:
                # Assume both spectra have the same frequency and use the one of the left operand
                frequency = self.frequency

            intensity = self.intensity / other.intensity

            if self.noise is None or other.noise is None:
                noise = None
            else:
                noise = intensity * numpy.sqrt(
                    numpy.square(self.noise / self.intensity)
                    + numpy.square(other.noise / other.intensity)
                )

            if self.flag is None or other.flag is None:
                flag = None
            else:
                flag = self.flag | other.flag

            return Spectrum(
                intensity=intensity, frequency=frequency, noise=noise, flag=flag
            )

        return NotImplemented

    def __rtruediv__(self, other: int | float) -> "Spectrum":
        if isinstance(other, int) or isinstance(other, float):
            if self.frequency is None:
                frequency = None
            else:
                frequency = self.frequency

            intensity = other / self.intensity
            if self.noise is None:
                noise = None
            else:
                noise = intensity * (self.noise / self.intensity)

            if self.flag is None:
                flag = None
            else:
                flag = self.flag

            return Spectrum(
                intensity=intensity, frequency=frequency, noise=noise, flag=flag
            )

        return NotImplemented

    @property
    def flagged(self) -> numpy.typing.NDArray[numpy.bool_]:
        if self.flag is None:
            loguru.logger.warning("This spectrum have no flags.")
            return numpy.full_like(self.intensity, False)

        return self.flag != Spectrum.FlagReason.NOT_FLAGGED.value

    def detect_signal(
        self, *, nadjacent: int, alpha: float, chunk_size: int = 1024
    ) -> numpy.typing.NDArray[numpy.bool_] | None:
        if self.noise is None:
            loguru.logger.warning("This spectrum has no noise.")
            return None

        if self.frequency is None:
            frequency = numpy.arange(self.intensity.size, dtype=numpy.float64)
        else:
            frequency = self.frequency

        # Here, we want to calculate for each point the minimum chi-square that
        # (2n + 1) adjacent points centered on the given point lies on a
        # straight line mx + c.
        # We define chi-squared = Sum_i (((mx_i + c) - y_i) / s_i)^2 and
        # minimizing it w.r.t. m and c gives a system of equation of the form:
        #   p * m + q * c = r
        #   u * m + v * c = w
        # where
        #   p = Sum_i (x_i**2 / s_i**2)
        #   q = Sum_i (x_i / s_i**2)
        #   r = Sum_i (x_i * y_i / s_i**2)
        #   u = q
        #   v = Sum_i (1 / s_i**2)
        #   w = Sum_i (y_i / s_i**2)
        # Solving the set of equation using Cramer's rule for m and c and
        # substitude them back to definition, we get
        # chi-squared = (
        #     sum_one_s2 * (sum_x2_s2 * sum_y2_s2 - sum_xy_s2 * sum_xy_s2)
        #     + sum_x_s2 * (sum_xy_s2 * sum_y_s2 - sum_x_s2 * sum_y2_s2)
        #     + sum_y_s2 * (sum_xy_s2 * sum_x_s2 - sum_y_s2 * sum_x2_s2)
        # ) / (sum_one_s2 * sum_x2_s2 - sum_x_s2 * sum_x_s2)
        size = frequency.size - 2 * nadjacent
        chi_squared = numpy.empty(size, dtype=float)
        for lb, ub in itertools.pairwise([*range(0, size, chunk_size), size]):
            x = frequency[lb : ub + 2 * nadjacent]
            y = self.intensity[lb : ub + 2 * nadjacent]
            s = self.noise[lb : ub + 2 * nadjacent]
            x = x - x.mean()
            y = y - y.mean()
            one_s2 = 1 / numpy.square(s)
            x_s2 = x * one_s2
            y_s2 = y * one_s2
            x2_s2 = x * x_s2
            xy_s2 = x * y_s2
            y2_s2 = y * y_s2

            width = 2 * nadjacent + 1
            sum_one_s2 = bottleneck.move_sum(one_s2, window=width)[2 * nadjacent :]
            sum_x_s2 = bottleneck.move_sum(x_s2, window=width)[2 * nadjacent :]
            sum_y_s2 = bottleneck.move_sum(y_s2, window=width)[2 * nadjacent :]
            sum_x2_s2 = bottleneck.move_sum(x2_s2, window=width)[2 * nadjacent :]
            sum_xy_s2 = bottleneck.move_sum(xy_s2, window=width)[2 * nadjacent :]
            sum_y2_s2 = bottleneck.move_sum(y2_s2, window=width)[2 * nadjacent :]

            chi_squared[lb:ub] = (
                sum_one_s2 * (sum_x2_s2 * sum_y2_s2 - sum_xy_s2 * sum_xy_s2)
                + sum_x_s2 * (sum_xy_s2 * sum_y_s2 - sum_x_s2 * sum_y2_s2)
                + sum_y_s2 * (sum_xy_s2 * sum_x_s2 - sum_y_s2 * sum_x2_s2)
            ) / (sum_one_s2 * sum_x2_s2 - sum_x_s2 * sum_x_s2)

        is_signal = numpy.where(chi_squared > scipy.special.chdtri(width, alpha))[0]

        res = numpy.zeros_like(self.intensity, dtype=bool)
        for i in range(width):
            res[i : size + i][is_signal] = True
        return res

    def flag_rfi(
        self, *, nadjacent: int = 3, alpha: float = 1e-6, chunk_size: int = 1024
    ):
        if self.flag is None:
            loguru.logger.warning("This spectrum have no flags.")
            return

        is_rfi = self.detect_signal(
            nadjacent=nadjacent, alpha=alpha, chunk_size=chunk_size
        )
        self.flag[is_rfi] = self.flag[is_rfi] | Spectrum.FlagReason.RFI.value

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
            nchannel = int(self.intensity.size * fraction)

        if nchannel > 0:
            self.flag[:nchannel] = (
                self.flag[:nchannel] | Spectrum.FlagReason.CHUNK_EDGES.value
            )
            self.flag[-nchannel:] = (
                self.flag[-nchannel:] | Spectrum.FlagReason.CHUNK_EDGES.value
            )

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
        frequency = (
            numpy.arange(fitted.size)
            if self.frequency is None
            else self.frequency[fitted]
        )
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
        return Spectrum(intensity=intensity, frequency=self.frequency, noise=noise)

    def split(self, min_length: int = 2) -> "list[Spectrum]":
        if self.flag is None:
            loguru.logger.warning("This spectrum have no flags.")
            return [self]

        spectrum_chunks: list[Spectrum] = []
        (split_positions,) = numpy.where(
            numpy.diff(numpy.pad(self.flagged, 1, constant_values=True))
        )
        for start, stop in split_positions.reshape(-1, 2):
            if stop - start < min_length:
                continue
            intensity = self.intensity[start:stop]
            frequency = (
                self.frequency[start:stop] if self.frequency is not None else None
            )
            noise = self.noise[start:stop] if self.noise is not None else None
            flag = self.flag[start:stop] if self.flag is not None else None
            spectrum_chunks.append(
                Spectrum(
                    intensity=intensity, frequency=frequency, noise=noise, flag=flag
                )
            )
        return spectrum_chunks


class SpectrumAggregator:

    class Transformer:

        def forward(
            self,
            index: numpy.typing.NDArray[numpy.floating]
            | numpy.typing.NDArray[numpy.integer],
        ) -> numpy.typing.NDArray[numpy.floating]:
            raise NotImplementedError

        def backward(
            self, value: numpy.typing.NDArray[numpy.floating]
        ) -> numpy.typing.NDArray[numpy.floating]:
            raise NotImplementedError

    class LinearTransformer(Transformer):
        step: float
        zero_point: float

        def __init__(self, step: float, zero_point: float = 0.0):
            self.step = step
            self.zero_point = zero_point

        def forward(
            self,
            index: numpy.typing.NDArray[numpy.floating]
            | numpy.typing.NDArray[numpy.integer],
        ) -> numpy.typing.NDArray[numpy.floating]:
            return index * self.step + self.zero_point

        def backward(
            self, value: numpy.typing.NDArray[numpy.floating]
        ) -> numpy.typing.NDArray[numpy.floating]:
            return (value - self.zero_point) / self.step

    class SpectrumSlice(typing.NamedTuple):
        start: int
        stop: int
        weighted_intensity: numpy.typing.NDArray[numpy.floating]
        weighted_variance: numpy.typing.NDArray[numpy.floating] | None
        weight: numpy.typing.NDArray[numpy.floating]

        def __eq__(self, other) -> bool:
            if isinstance(other, SpectrumAggregator.SpectrumSlice):
                return self is other
            return NotImplemented

    transformer: Transformer
    compute_noise: bool
    noise_weighted: bool
    spectrum_slices: SortedList

    def __init__(
        self,
        transformer: Transformer,
        compute_noise: bool = True,
        noise_weighted: bool = True,
    ):
        self.transformer = transformer
        self.compute_noise = compute_noise
        self.noise_weighted = noise_weighted
        self.spectrum_slices = SortedList(
            key=lambda spectrum_slice: spectrum_slice.start
        )

    def _make_spectrum_slice(self, spectrum: Spectrum) -> SpectrumSlice | None:
        if spectrum.frequency is None:
            loguru.logger.error("This spectrum does not have frequency.")
            return None

        if spectrum.noise is None and (self.noise_weighted or self.compute_noise):
            loguru.logger.error("This spectrum does not have frequency.")
            return None

        start = int(numpy.ceil(self.transformer.backward(spectrum.frequency[0])))
        stop = int(numpy.ceil(self.transformer.backward(spectrum.frequency[-1])))
        if start >= stop:
            return None

        frequency = self.transformer.forward(numpy.arange(start, stop))
        intensity = numpy.interp(
            frequency,
            spectrum.frequency,
            spectrum.intensity,
            left=0.0,
            right=0.0,
        )

        if self.noise_weighted or self.compute_noise:
            assert spectrum.noise is not None
            variance = numpy.square(
                numpy.interp(
                    frequency,
                    spectrum.frequency,
                    spectrum.noise,
                    left=numpy.inf,
                    right=numpy.inf,
                )
            )
        else:
            variance = None

        if self.noise_weighted:
            assert variance is not None
            weight = 1.0 / variance
        else:
            weight = numpy.ones_like(spectrum.frequency)

        if self.compute_noise:
            assert variance is not None
            spectrum_slice = self.SpectrumSlice(
                start=start,
                stop=stop,
                weighted_intensity=weight * intensity,
                weighted_variance=numpy.square(weight) * variance,
                weight=weight,
            )
        else:
            spectrum_slice = self.SpectrumSlice(
                start=start,
                stop=stop,
                weighted_intensity=weight * intensity,
                weighted_variance=None,
                weight=weight,
            )

        return spectrum_slice

    def _clean_range(self, start: int | None = None, stop: int | None = None):
        if not self.spectrum_slices:
            return
        if start is None:
            start = self.spectrum_slices[0].start
        if stop is None:
            stop = self.spectrum_slices[-1].stop

        grouped_slices: list[list[SpectrumAggregator.SpectrumSlice]] = []
        for spectrum_slice in self.spectrum_slices.irange_key(
            max_key=stop, reverse=True
        ):
            if spectrum_slice.stop < start:
                break
            if not grouped_slices or spectrum_slice.stop < grouped_slices[-1][-1].start:
                grouped_slices.append([])
            grouped_slices[-1].append(spectrum_slice)

        for slice_group in grouped_slices:
            if len(slice_group) == 1:
                continue

            min_start = min(slc.start for slc in slice_group)
            max_stop = max(slc.stop for slc in slice_group)
            for slc in slice_group:
                if slc.start == min_start and slc.stop == max_stop:
                    combined_slice = slc
                    break
            else:
                combined_slice = self.SpectrumSlice(
                    start=min_start,
                    stop=max_stop,
                    weighted_intensity=numpy.zeros(max_stop - min_start),
                    weighted_variance=numpy.zeros(max_stop - min_start)
                    if self.compute_noise
                    else None,
                    weight=numpy.zeros(max_stop - min_start),
                )
                self.spectrum_slices.add(combined_slice)
            for slc in slice_group:
                if slc is combined_slice:
                    continue
                interval = slice(slc.start - min_start, slc.stop - min_start)
                combined_slice.weighted_intensity[interval] += slc.weighted_intensity
                if self.compute_noise:
                    assert combined_slice.weighted_variance is not None
                    assert slc.weighted_variance is not None
                    combined_slice.weighted_variance[interval] += slc.weighted_variance
                combined_slice.weight[interval] += slc.weight
                self.spectrum_slices.discard(slc)

    def merge(self, spectrum: Spectrum):
        dirty_slices = [*map(self._make_spectrum_slice, spectrum.split())]
        clean_slices = [slc for slc in dirty_slices if slc is not None]
        if len(clean_slices) != len(dirty_slices):
            loguru.logger.warning("Some spectrum slices are not merged")
        if not clean_slices:
            return

        self.spectrum_slices.update(clean_slices)
        self._clean_range(
            start=clean_slices[0].start,
            stop=clean_slices[-1].stop,
        )

    def get_spectrum(self) -> Spectrum:
        total_length = (
            sum(map(lambda slc: slc.stop - slc.start + 1, self.spectrum_slices)) - 1
        )
        frequency = numpy.full(total_length, numpy.nan)
        intensity = numpy.full(total_length, numpy.nan)
        noise = numpy.full(total_length, numpy.nan) if self.compute_noise else None
        filled_length = 0
        for spectrum_slice in self.spectrum_slices:
            current_length = spectrum_slice.stop - spectrum_slice.start
            interval = slice(filled_length, filled_length + current_length)
            frequency[interval] = self.transformer.forward(
                numpy.arange(spectrum_slice.start, spectrum_slice.stop)
            )
            intensity[interval] = (
                spectrum_slice.weighted_intensity / spectrum_slice.weight
            )
            if self.compute_noise:
                assert noise is not None
                noise[interval] = (
                    numpy.sqrt(spectrum_slice.weighted_variance) / spectrum_slice.weight
                )
            filled_length += current_length + 1
        return Spectrum(intensity=intensity, frequency=frequency, noise=noise)
