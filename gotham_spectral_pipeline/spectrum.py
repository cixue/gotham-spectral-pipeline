import collections
import dataclasses
import enum
import functools
import itertools
import pathlib
import typing
from typing_extensions import Self

import astropy.timeseries  # type: ignore
import bottleneck  # type: ignore
import loguru
import numpy
import numpy.polynomial.polynomial as poly
import numpy.typing
import scipy.special  # type: ignore
from sortedcontainers import SortedList  # type: ignore

from .utils import searchsorted_nearest

__all__ = [
    "Spectrum",
    "Exposure",
    "SpectrumAggregator",
    "ExposureAggregator",
]

Baseline = typing.Callable[
    [numpy.typing.NDArray[numpy.floating]], numpy.typing.NDArray[numpy.floating]
]
BaselineSupplementaryInfo = dict[str, typing.Any]
_SpectrumLike = typing.TypeVar("_SpectrumLike", bound="SpectrumLike")


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
    return numpy.max(
        numpy.fabs(_centered_move_mean(intensity / noise, half_moving_window))
    )


class SpectrumLike:
    _fields: dict[str, numpy.typing.NDArray | None]

    def __init__(
        self, **fields: tuple[numpy.typing.ArrayLike | None, numpy.typing.DTypeLike]
    ):
        self._fields = dict()
        for key, (value, dtype) in fields.items():
            if value is not None:
                value = numpy.array(value, dtype=dtype)
            self._fields[key] = value
            self.__setattr__(key, value)

        shapes = {
            ndarray.shape for ndarray in self._fields.values() if ndarray is not None
        }
        if len(shapes) != 1:
            loguru.logger.error("Shapes of input arrays are not identical.")
            return

    def sort_by(self, name: str):
        target_field = self._fields.get(name)
        if target_field is None:
            loguru.logger.error(f"Field {name} does not exist.")
            return

        is_sorted = lambda a: numpy.all(a[1:] >= a[:-1])

        (nan_positions,) = numpy.where(
            numpy.diff(numpy.pad(numpy.isnan(target_field), 1, constant_values=True))
        )
        for start, stop in nan_positions.reshape(-1, 2):
            current = slice(start, stop)

            if is_sorted(target_field[current]):
                continue

            if is_sorted(target_field[current][::-1]):
                for value in self._fields.values():
                    if value is not None:
                        value[current] = value[current][::-1]
                continue

            sortidx = target_field[current].argsort()
            for value in self._fields.values():
                if value is not None:
                    value[current] = value[current][sortidx]

    def split(self, min_length: int = 2) -> list[Self]:
        raise NotImplementedError

    def to_npz(self, path: pathlib.Path):
        output = {
            key: value for key, value in self._fields.items() if value is not None
        }
        numpy.savez_compressed(path, **output)

    @classmethod
    def from_npz(cls, path: pathlib.Path) -> Self:
        output = numpy.load(path)
        return cls(**output)

    def to_txt(
        self,
        path: pathlib.Path,
        columns: list[str] | None = None,
    ):
        if columns is None:
            columns = list(self._fields.keys())
        output: list[numpy.typing.NDArray] = []
        for column in columns:
            field = self._fields.get(column)
            if field is None:
                loguru.logger.error(f"This spectrum does not have {column}")
                return
            output.append(field)
        numpy.savetxt(path, numpy.stack(output, axis=-1), header=f"{' '.join(columns)}")

    @classmethod
    def from_txt(cls, path: pathlib.Path) -> Self:
        with open(path, mode="r") as fin:
            header = fin.readline()
            columns = header.lstrip("# ").split()
            output = numpy.loadtxt(fin)
        return cls(**dict(zip(columns, output)))


class Spectrum(SpectrumLike):
    frequency: numpy.typing.NDArray[numpy.floating] | None
    intensity: numpy.typing.NDArray[numpy.floating]
    noise: numpy.typing.NDArray[numpy.floating] | None
    flag: numpy.typing.NDArray[numpy.signedinteger] | None

    class FlagReason(enum.Flag):
        NOT_FLAGGED = 0
        NAN = enum.auto()
        CHUNK_EDGES = enum.auto()
        FREQUENCY_DOMAIN_RFI = enum.auto()
        TIME_DOMAIN_RFI = enum.auto()
        SIGNAL = enum.auto()
        VALID_DATA = enum.auto()

    def __init__(
        self,
        *,
        intensity: numpy.typing.ArrayLike,
        frequency: numpy.typing.ArrayLike | None = None,
        noise: numpy.typing.ArrayLike | None = None,
        flag: numpy.typing.ArrayLike | None = None,
    ):
        super().__init__(
            intensity=(intensity, numpy.float64),
            frequency=(frequency, numpy.float64),
            noise=(noise, numpy.float64),
            flag=(flag, numpy.int32),
        )

        if self.frequency is not None:
            self.sort_by("frequency")

    def __neg__(self) -> "Spectrum":
        if self.frequency is None:
            frequency = None
        else:
            frequency = self.frequency

        intensity = -self.intensity

        if self.noise is None:
            noise = None
        else:
            noise = self.noise

        if self.flag is None:
            flag = None
        else:
            flag = self.flag

        return Spectrum(
            intensity=intensity, frequency=frequency, noise=noise, flag=flag
        )

    def __add__(self, other: "int | float | Spectrum") -> "Spectrum":
        if isinstance(other, int) or isinstance(other, float):
            if self.frequency is None:
                frequency = None
            else:
                frequency = self.frequency

            intensity = self.intensity + other

            if self.noise is None:
                noise = None
            else:
                noise = self.noise

            if self.flag is None:
                flag = None
            else:
                flag = self.flag

            return Spectrum(
                intensity=intensity, frequency=frequency, noise=noise, flag=flag
            )

        if isinstance(other, Spectrum):
            intensity = self.intensity + other.intensity

            # Assume both spectra have the same frequency and use any one of them. Prefer the one of the left operand.
            if self.frequency is not None:
                frequency = self.frequency
            elif other.frequency is not None:
                frequency = other.frequency
            else:
                frequency = None

            if self.noise is not None and other.noise is not None:
                noise = numpy.sqrt(numpy.square(self.noise) + numpy.square(other.noise))
            elif self.noise is not None:
                noise = self.noise
            elif other.noise is not None:
                noise = other.noise
            else:
                noise = None

            if self.flag is not None and other.flag is not None:
                flag = self.flag | other.flag
            elif self.flag is not None:
                flag = self.flag
            elif other.flag is not None:
                flag = other.flag
            else:
                flag = None

            return Spectrum(
                intensity=intensity, frequency=frequency, noise=noise, flag=flag
            )

        return NotImplemented

    def __radd__(self, other: int | float) -> "Spectrum":
        return self + other

    def __sub__(self, other: "int | float | Spectrum") -> "Spectrum":
        return self + (-other)

    def __rsub__(self, other: int | float) -> "Spectrum":
        return -self + other

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
            # Assume both spectra have the same frequency and use any one of them. Prefer the one of the left operand.
            if self.frequency is not None:
                frequency = self.frequency
            elif other.frequency is not None:
                frequency = other.frequency
            else:
                frequency = None

            intensity = self.intensity * other.intensity

            if self.noise is not None and other.noise is not None:
                noise = intensity * numpy.sqrt(
                    numpy.square(self.noise / self.intensity)
                    + numpy.square(other.noise / other.intensity)
                )
            elif self.noise is not None:
                noise = self.noise * other.intensity
            elif other.noise is not None:
                noise = other.noise * self.intensity
            else:
                noise = None

            if self.flag is not None and other.flag is not None:
                flag = self.flag | other.flag
            elif self.flag is not None:
                flag = self.flag
            elif other.flag is not None:
                flag = other.flag
            else:
                flag = None

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
            # Assume both spectra have the same frequency and use any one of them. Prefer the one of the left operand.
            if self.frequency is not None:
                frequency = self.frequency
            elif other.frequency is not None:
                frequency = other.frequency
            else:
                frequency = None

            intensity = self.intensity / other.intensity

            if self.noise is not None and other.noise is not None:
                noise = intensity * numpy.sqrt(
                    numpy.square(self.noise / self.intensity)
                    + numpy.square(other.noise / other.intensity)
                )
            elif self.noise is not None:
                noise = self.noise / other.intensity
            elif other.noise is not None:
                noise = intensity * (other.noise / other.intensity)
            else:
                noise = None

            if self.flag is not None and other.flag is not None:
                flag = self.flag | other.flag
            elif self.flag is not None:
                flag = self.flag
            elif other.flag is not None:
                flag = other.flag
            else:
                flag = None

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

    def flagged(
        self, flags: "Spectrum.FlagReason | None" = None
    ) -> numpy.typing.NDArray[numpy.bool_]:
        if self.flag is None:
            loguru.logger.warning("This spectrum does not have flags.")
            return numpy.full_like(self.intensity, False)
        if flags is None:
            return self.flag != 0
        return (self.flag & flags.value) != 0

    def _add_flag_array(self):
        self.flag = numpy.full_like(
            self.intensity, Spectrum.FlagReason.NOT_FLAGGED.value, dtype=numpy.int32
        )

    def detect_signal(
        self,
        *,
        nadjacent: int | dict[typing.Literal["baseline", "chisq"], int],
        alpha: float,
        chunk_size: int = 1024,
        mask: numpy.typing.NDArray[numpy.bool_] | None = None,
    ) -> numpy.typing.NDArray[numpy.bool_] | None:
        if self.noise is None:
            loguru.logger.warning("This spectrum does not have noise.")
            return None

        if isinstance(nadjacent, dict):
            nbaseline = nadjacent["baseline"]
            nchisq = nadjacent["chisq"]
        else:
            nbaseline = nchisq = nadjacent

        if nbaseline < nchisq:
            loguru.logger.error(
                "nadjacent used for computing chisq must not be larger than that for computing baseline."
            )
            return None

        if self.frequency is None:
            frequency = numpy.arange(self.intensity.size, dtype=numpy.float64)
        else:
            frequency = self.frequency
        intensity = self.intensity
        noise = self.noise

        if mask is not None:
            frequency = frequency[mask]
            intensity = intensity[mask]
            noise = noise[mask]

        # Here, we want to calculate for each point the minimum chi-square that
        # (2n + 1) adjacent points centered on the given point lies on a
        # straight line mx + c.
        # We define chi-squared = Sum_i (((mx_i + c) - y_i) / s_i)^2 and
        # minimizing it w.r.t. m and c gives a system of equation of the form:
        #   sxx * m + sx * c = sxy
        #   sx  * m + s  * c = sy
        # where
        #   sxx = Sum_i (x_i**2 / s_i**2)
        #   sxy = Sum_i (x_i * y_i / s_i**2)
        #   syy = Sum_i (y_i**2 / s_i**2)
        #   sx = Sum_i (x_i / s_i**2)
        #   sy = Sum_i (y_i / s_i**2)
        #   s = Sum_i (1 / s_i**2)
        # Solving the set of equation using Cramer's rule for m and c, we get
        #   d = sxx * s - sx**2
        #   dm = sxy * s - sx * sy
        #   dc = sxx * sy - sx * sxy
        #   m = dm / d
        #   c = dc / d
        # Substitude them back to definition, we get
        #   chisq = m**2 * sxx + c**2 * s + syy + 2 * (m * c * sx - m * sxy - c * sy)
        # Simplifying it gives
        #   chisq = (
        #       s * (sxx * syy - sxy * sxy)
        #       + sx * (sxy * sy - sx * syy)
        #       + sy * (sxy * sx - sy * sxx)
        #   ) / (s * sxx - sx * sx)
        width_baseline = 2 * nbaseline + 1
        width_chisq = 2 * nchisq + 1
        size = frequency.size - width_baseline + 1
        chisq = numpy.empty(size, dtype=float)
        for lb, ub in itertools.pairwise([*range(0, size, chunk_size), size]):
            move_sum = lambda arr, window: bottleneck.move_sum(arr, window=window)[
                window - 1 :
            ]

            x = frequency[lb : ub + width_baseline - 1]
            y = intensity[lb : ub + width_baseline - 1]
            s = noise[lb : ub + width_baseline - 1]
            x = x - x.mean()
            y = y - y.mean()
            one_s2 = 1 / numpy.square(s)
            x_s2 = x * one_s2
            y_s2 = y * one_s2
            x2_s2 = x * x_s2
            xy_s2 = x * y_s2
            y2_s2 = y * y_s2

            s = move_sum(one_s2, window=width_baseline)
            sx = move_sum(x_s2, window=width_baseline)
            sy = move_sum(y_s2, window=width_baseline)
            sxx = move_sum(x2_s2, window=width_baseline)
            sxy = move_sum(xy_s2, window=width_baseline)
            syy = move_sum(y2_s2, window=width_baseline)

            if nbaseline == nchisq:
                chisq[lb:ub] = (
                    s * (sxx * syy - sxy * sxy)
                    + sx * (sxy * sy - sx * syy)
                    + sy * (sxy * sx - sy * sxx)
                ) / (s * sxx - sx * sx)
            else:
                d = sxx * s - sx * sx
                dm = sxy * s - sx * sy
                dc = sxx * sy - sx * sxy
                m = dm / d
                c = dc / d

                ndiff = nbaseline - nchisq
                s = move_sum(one_s2[ndiff:-ndiff], window=width_chisq)
                sx = move_sum(x_s2[ndiff:-ndiff], window=width_chisq)
                sy = move_sum(y_s2[ndiff:-ndiff], window=width_chisq)
                sxx = move_sum(x2_s2[ndiff:-ndiff], window=width_chisq)
                sxy = move_sum(xy_s2[ndiff:-ndiff], window=width_chisq)
                syy = move_sum(y2_s2[ndiff:-ndiff], window=width_chisq)

                chisq[lb:ub] = (
                    m * m * sxx + c * c * s + syy + 2 * (m * c * sx - m * sxy - c * sy)
                )

        is_signal = numpy.where(chisq > scipy.special.chdtri(width_chisq, alpha))[0]

        res = numpy.zeros_like(intensity, dtype=bool)
        for i in range(width_chisq):
            ndiff = nbaseline - nchisq
            res[ndiff + i : ndiff + size + i][is_signal] = True
        if mask is None:
            return res
        res_unmasked = numpy.zeros_like(self.intensity, dtype=bool)
        res_unmasked[mask] = res
        return res_unmasked

    def flag_valid_data(self) -> Self:
        if self.flag is None:
            self._add_flag_array()
        assert self.flag is not None

        is_valid = ~self.flagged(
            Spectrum.FlagReason.NAN
            | Spectrum.FlagReason.CHUNK_EDGES
            | Spectrum.FlagReason.FREQUENCY_DOMAIN_RFI
            | Spectrum.FlagReason.TIME_DOMAIN_RFI
        )
        self.flag[is_valid] |= Spectrum.FlagReason.VALID_DATA.value
        return self

    def flag_signal(
        self,
        *,
        nadjacent: int | dict[typing.Literal["baseline", "chisq"], int] = 31,
        alpha: float = 1e-6,
        chunk_size: int = 1024,
    ) -> Self:
        if self.flag is None:
            self._add_flag_array()
        assert self.flag is not None

        is_signal = self.detect_signal(
            nadjacent=nadjacent,
            alpha=alpha,
            chunk_size=chunk_size,
            mask=~self.flagged(
                Spectrum.FlagReason.FREQUENCY_DOMAIN_RFI
                | Spectrum.FlagReason.TIME_DOMAIN_RFI
            ),
        )
        if is_signal is None:
            return self
        self.flag[is_signal] |= Spectrum.FlagReason.SIGNAL.value
        return self

    def flag_time_domain_rfi(
        self,
        spectrum_metadata: dict,
        *,
        nadjacent: int | dict[typing.Literal["baseline", "chisq"], int] = 63,
        alpha: float = 1e-6,
        chunk_size: int = 1024,
    ) -> Self:
        if self.flag is None:
            self._add_flag_array()
        assert self.flag is not None

        temporal_difference = (
            spectrum_metadata["sig_calon"] - spectrum_metadata["sig_caloff"]
        ) - (spectrum_metadata["ref_calon"] - spectrum_metadata["ref_caloff"])
        is_rfi = temporal_difference.detect_signal(
            nadjacent=nadjacent, alpha=alpha, chunk_size=chunk_size
        )
        if is_rfi is None:
            return self
        self.flag[is_rfi] |= Spectrum.FlagReason.TIME_DOMAIN_RFI.value
        return self

    def flag_frequency_domain_rfi(
        self,
        *,
        nadjacent: int | dict[typing.Literal["baseline", "chisq"], int] = 1,
        alpha: float = 1e-6,
        chunk_size: int = 1024,
    ) -> Self:
        if self.flag is None:
            self._add_flag_array()
        assert self.flag is not None

        is_rfi = self.detect_signal(
            nadjacent=nadjacent, alpha=alpha, chunk_size=chunk_size
        )
        if is_rfi is None:
            return self
        self.flag[is_rfi] |= Spectrum.FlagReason.FREQUENCY_DOMAIN_RFI.value
        return self

    def flag_head_tail(
        self, *, fraction: float | None = None, nchannel: int | None = None
    ) -> Self:
        if self.flag is None:
            self._add_flag_array()
        assert self.flag is not None

        if nchannel is None:
            if fraction is None:
                loguru.logger.error(
                    "Either but not both fraction or nchannel should be set."
                )
                return self
            nchannel = int(self.intensity.size * fraction)

        if nchannel > 0:
            self.flag[:nchannel] |= Spectrum.FlagReason.CHUNK_EDGES.value
            self.flag[-nchannel:] |= Spectrum.FlagReason.CHUNK_EDGES.value
        return self

    def flag_nan(self) -> Self:
        if self.flag is None:
            self._add_flag_array()
        assert self.flag is not None

        is_nan = numpy.isnan(self.intensity)
        if self.frequency is not None:
            is_nan |= numpy.isnan(self.frequency)
        if self.noise is not None:
            is_nan |= numpy.isnan(self.noise)
        self.flag[is_nan] |= Spectrum.FlagReason.NAN.value
        return self

    def fit_baseline(
        self,
        method: typing.Literal["hybrid", "polynomial", "lomb-scargle"] = "hybrid",
        residual_threshold: float | None = None,
        residual_half_moving_window: int = 512,
        polynomial_options: dict[str, typing.Any] = {},
        lomb_scargle_options: dict[str, typing.Any] = {},
    ) -> tuple[Baseline, BaselineSupplementaryInfo] | None:
        fitted = ~self.flagged(~Spectrum.FlagReason.VALID_DATA)
        frequency = (
            numpy.arange(fitted.size, dtype=numpy.float64)
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
    ) -> "Spectrum":
        has_valid_data = self.flagged(Spectrum.FlagReason.VALID_DATA)
        count = _centered_move_sum(has_valid_data, half_moving_window)
        intensity = (
            _centered_move_sum(
                numpy.where(has_valid_data, self.intensity, numpy.nan),
                half_moving_window,
            )
            / count
        )
        noise = (
            None
            if self.noise is None
            else numpy.sqrt(
                _centered_move_sum(
                    numpy.where(has_valid_data, self.noise**2, numpy.nan),
                    half_moving_window,
                )
            )
            / count
        )
        return Spectrum(intensity=intensity, frequency=self.frequency, noise=noise)

    def split(self, min_length: int = 2) -> "list[Spectrum]":
        if self.flag is None:
            loguru.logger.warning("This spectrum does not have flags.")
            return [self]

        spectrum_chunks: list[Spectrum] = []
        (split_positions,) = numpy.where(
            numpy.diff(
                numpy.pad(
                    ~self.flagged(Spectrum.FlagReason.VALID_DATA),
                    1,
                    constant_values=True,
                )
            )
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

    def copy_flags(self, flags: "Spectrum.FlagReason", spectrum: "Spectrum") -> Self:
        if spectrum.flag is None:
            loguru.logger.warning("This spectrum does not have flags to be copied.")
            return self
        if self.frequency is None or spectrum.frequency is None:
            loguru.logger.warning(
                "Either or both of the spectra does not have frequency."
            )
            return self

        if self.flag is None:
            self._add_flag_array()
        assert self.flag is not None

        nearest, out_of_bound = searchsorted_nearest(spectrum.frequency, self.frequency)
        copied_flag = spectrum.flag[nearest] & flags.value
        copied_flag[out_of_bound] = Spectrum.FlagReason.NOT_FLAGGED.value
        self.flag &= (~flags).value
        self.flag |= copied_flag
        return self

    def from_callable(self, callable: typing.Callable) -> "Spectrum":
        return Spectrum(intensity=callable(self.frequency), frequency=self.frequency)


class Exposure(SpectrumLike):
    exposure: numpy.typing.NDArray[numpy.floating]
    frequency: numpy.typing.NDArray[numpy.floating]

    def __init__(
        self,
        *,
        exposure: numpy.typing.ArrayLike,
        frequency: numpy.typing.ArrayLike,
    ):
        super().__init__(
            exposure=(exposure, numpy.float64),
            frequency=(frequency, numpy.float64),
        )

        if self.frequency is not None:
            self.sort_by("frequency")

    def split(self, min_length: int = 2) -> "list[Exposure]":
        exposure_chunks: list[Exposure] = []
        (split_positions,) = numpy.where(
            numpy.diff(numpy.pad(numpy.isnan(self.frequency), 1, constant_values=True))
        )
        for start, stop in split_positions.reshape(-1, 2):
            if stop - start < min_length:
                continue
            exposure = self.exposure[start:stop]
            frequency = self.frequency[start:stop]
            exposure_chunks.append(Exposure(exposure=exposure, frequency=frequency))
        return exposure_chunks


class Aggregator(typing.Generic[_SpectrumLike]):

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

    @dataclasses.dataclass
    class SpectrumSlice:
        start: int
        stop: int
        buffer_start: int
        buffer_stop: int
        data: dict[
            str,
            numpy.typing.NDArray[numpy.floating] | numpy.typing.NDArray[numpy.integer],
        ]

        def __eq__(self, other) -> bool:
            if isinstance(other, Aggregator.SpectrumSlice):
                return self is other
            return NotImplemented

        @classmethod
        def new_empty_slice(cls, start: int, stop: int) -> Self:
            length = stop - start
            buffer_start = max((start + stop) // 2 - length, 0)
            buffer_stop = buffer_start + 2 * length

            return cls(
                start=start,
                stop=stop,
                buffer_start=buffer_start,
                buffer_stop=buffer_stop,
                data=dict(),
            )

    spectrum_class: type
    aligned_by: str
    transformer: Transformer
    spectrum_slices: SortedList
    options: dict

    def __init__(
        self,
        spectrum_class: type[_SpectrumLike],
        aligned_by: str,
        transformer: Transformer,
        options: dict | None = None,
    ):
        self.spectrum_class = spectrum_class
        self.aligned_by = aligned_by
        self.transformer = transformer
        self.spectrum_slices = SortedList(
            key=lambda spectrum_slice: spectrum_slice.start
        )
        self.options = dict() if options is None else options

    def _init_spectrum_slice(self, spectrum_slice: SpectrumSlice):
        raise NotImplementedError

    def _fill_spectrum_slice(
        self,
        spectrum_slice: SpectrumSlice,
        spectrum: _SpectrumLike,
        interval: slice,
        aligned_value: numpy.typing.NDArray[numpy.floating],
    ):
        raise NotImplementedError

    def _add_spectrum_slices(
        self,
        mutable: SpectrumSlice,
        constant: SpectrumSlice,
        mutable_interval: slice,
        constant_interval: slice,
    ):
        raise NotImplementedError

    def _make_spectrum_slice(
        self, spectrum: _SpectrumLike, init_only: bool = False
    ) -> SpectrumSlice | None:
        aligned_field = spectrum._fields.get(self.aligned_by)
        if aligned_field is None:
            loguru.logger.error(
                f"Required field '{self.aligned_by}' but it is not available."
            )
            return None

        start = int(numpy.ceil(self.transformer.backward(aligned_field[0])))
        stop = int(numpy.floor(self.transformer.backward(aligned_field[-1]))) + 1
        if start >= stop:
            return None
        spectrum_slice = self.SpectrumSlice.new_empty_slice(start, stop)
        self._init_spectrum_slice(spectrum_slice)

        if init_only:
            return spectrum_slice

        interval = slice(
            start - spectrum_slice.buffer_start, stop - spectrum_slice.buffer_start
        )
        aligned_value = self.transformer.forward(numpy.arange(start, stop))
        self._fill_spectrum_slice(spectrum_slice, spectrum, interval, aligned_value)
        return spectrum_slice

    def _clean_slices(
        self,
        new_spectrum_slices: list[SpectrumSlice] | None = None,
        owned: bool = False,
    ):
        if not self.spectrum_slices:
            return
        if new_spectrum_slices is None:
            new_spectrum_slices = self.spectrum_slices
        start = new_spectrum_slices[0].start
        stop = new_spectrum_slices[-1].stop

        first_slice_before_start = max(
            self.spectrum_slices.bisect_key_left(start) - 1, 0
        )
        grouped_slices: list[list[SpectrumAggregator.SpectrumSlice]] = []
        last_group_stop_max = -1
        for spectrum_slice in self.spectrum_slices.islice(
            start=first_slice_before_start
        ):
            if spectrum_slice.start > last_group_stop_max:
                if spectrum_slice.start > stop:
                    break
                grouped_slices.append([])
                last_group_stop_max = -1
            grouped_slices[-1].append(spectrum_slice)
            last_group_stop_max = max(last_group_stop_max, spectrum_slice.stop)

        for slice_group in grouped_slices:
            if len(slice_group) == 1 and (
                owned or slice_group[0] not in new_spectrum_slices
            ):
                continue

            min_start = min(slc.start for slc in slice_group)
            max_stop = max(slc.stop for slc in slice_group)
            eligible_reused_slices = [
                slc
                for slc in slice_group
                if (owned or slc not in new_spectrum_slices)
                and slc.buffer_start <= min_start
                and slc.buffer_stop >= max_stop
            ]
            if eligible_reused_slices:
                combined_slice = max(
                    eligible_reused_slices,
                    key=lambda slc: slc.buffer_stop - slc.buffer_start,
                )
                self.spectrum_slices.discard(combined_slice)
                combined_slice.start = min_start
                combined_slice.stop = max_stop
            else:
                combined_slice = self.SpectrumSlice.new_empty_slice(min_start, max_stop)
                self._init_spectrum_slice(combined_slice)
            self.spectrum_slices.add(combined_slice)
            for slc in slice_group:
                if slc is combined_slice:
                    continue
                combined_slice_interval = slice(
                    slc.start - combined_slice.buffer_start,
                    slc.stop - combined_slice.buffer_start,
                )
                current_interval = slice(
                    slc.start - slc.buffer_start, slc.stop - slc.buffer_start
                )
                self._add_spectrum_slices(
                    combined_slice, slc, combined_slice_interval, current_interval
                )
                self.spectrum_slices.discard(slc)

    def merge(
        self,
        spectrum: "_SpectrumLike | Aggregator[_SpectrumLike]",
        init_only: bool = False,
    ) -> Self:
        if isinstance(spectrum, self.spectrum_class):
            assert isinstance(spectrum, SpectrumLike)
            spectrum_slice = self._make_spectrum_slice(spectrum, init_only=init_only)  # type: ignore
            if spectrum_slice is None:
                return self
            clean_slices = [spectrum_slice]
            owned = True
        elif isinstance(spectrum, type(self)):
            clean_slices = spectrum.spectrum_slices
            owned = False
        else:
            loguru.logger.error(f"{type(self)}: Unsupported type {type(spectrum)}")
            return self

        if not clean_slices:
            return self

        self.spectrum_slices.update(clean_slices)
        self._clean_slices(new_spectrum_slices=clean_slices, owned=owned)
        return self

    def merge_all(
        self, spectra: "typing.Iterable[_SpectrumLike | Aggregator[_SpectrumLike]]"
    ) -> Self:
        for spectrum in spectra:
            self.merge(spectrum)
        return self

    def get_spectrum(self) -> _SpectrumLike:
        raise NotImplementedError


class SpectrumAggregator(Aggregator[Spectrum]):

    def __init__(self, *args, **kwargs):
        super().__init__(Spectrum, "frequency", *args, **kwargs)

    def _init_spectrum_slice(self, spectrum_slice: Aggregator.SpectrumSlice):
        buffer_length = spectrum_slice.buffer_stop - spectrum_slice.buffer_start
        spectrum_slice.data["weight"] = numpy.zeros(buffer_length, dtype=numpy.float64)
        spectrum_slice.data["weighted_intensity"] = numpy.zeros(
            buffer_length, dtype=numpy.float64
        )
        spectrum_slice.data["weighted_variance"] = numpy.zeros(
            buffer_length, dtype=numpy.float64
        )
        spectrum_slice.data["flag"] = numpy.full(
            buffer_length, Spectrum.FlagReason.NOT_FLAGGED.value, dtype=numpy.int32
        )

    def _fill_spectrum_slice(
        self,
        spectrum_slice: Aggregator.SpectrumSlice,
        spectrum: Spectrum,
        interval: slice,
        aligned_value: numpy.typing.NDArray[numpy.floating],
    ):
        assert spectrum.frequency is not None
        assert spectrum.noise is not None
        assert spectrum.flag is not None

        nearest, out_of_bound = searchsorted_nearest(spectrum.frequency, aligned_value)
        intensity = spectrum.intensity[nearest]
        intensity[out_of_bound] = 0.0
        noise = spectrum.noise[nearest]
        noise[out_of_bound] = numpy.inf

        variance = numpy.square(noise)
        weight = 1 / variance

        spectrum_slice.data["weight"][interval] = weight
        spectrum_slice.data["weighted_intensity"][interval] = weight * intensity
        spectrum_slice.data["weighted_variance"][interval] = (
            numpy.square(weight) * variance
        )
        spectrum_slice.data["flag"][interval] = spectrum.flag[nearest]
        flagged = (spectrum_slice.data["flag"][interval] & Spectrum.FlagReason.VALID_DATA.value) == 0  # type: ignore
        spectrum_slice.data["weight"][interval][flagged] = 0.0
        spectrum_slice.data["weighted_intensity"][interval][flagged] = 0.0
        spectrum_slice.data["weighted_variance"][interval][flagged] = 0.0

    def _add_spectrum_slices(
        self,
        mutable: Aggregator.SpectrumSlice,
        constant: Aggregator.SpectrumSlice,
        mutable_interval: slice,
        constant_interval: slice,
    ):
        correlated_variance = None
        if self.options.get("include_correlation", False):
            overlapped = (
                (
                    mutable.data["flag"][mutable_interval]
                    & Spectrum.FlagReason.VALID_DATA.value  # type: ignore
                    != 0
                )
                & (
                    mutable.data["flag"][mutable_interval]
                    & Spectrum.FlagReason.SIGNAL.value  # type: ignore
                    == 0
                )
                & (
                    constant.data["flag"][constant_interval]
                    & Spectrum.FlagReason.VALID_DATA.value  # type: ignore
                    != 0
                )
                & (
                    constant.data["flag"][constant_interval]
                    & Spectrum.FlagReason.SIGNAL.value  # type: ignore
                    == 0
                )
            )
            if overlapped.sum() != 0:
                mutable_overlapped = mutable.data["weighted_intensity"][
                    mutable_interval
                ][overlapped] / numpy.sqrt(
                    mutable.data["weighted_variance"][mutable_interval][overlapped]
                )
                constant_overlapped = constant.data["weighted_intensity"][
                    constant_interval
                ][overlapped] / numpy.sqrt(
                    constant.data["weighted_variance"][constant_interval][overlapped]
                )
                rho = numpy.nanmean(
                    (mutable_overlapped - numpy.nanmean(mutable_overlapped))
                    * (constant_overlapped - numpy.nanmean(constant_overlapped))
                )
                correlated_variance = (
                    2
                    * rho
                    * numpy.sqrt(
                        mutable.data["weighted_variance"][mutable_interval]
                        * constant.data["weighted_variance"][constant_interval]
                    )
                )

        mutable.data["weight"][mutable_interval] += constant.data["weight"][
            constant_interval
        ]
        mutable.data["weighted_intensity"][mutable_interval] += constant.data[
            "weighted_intensity"
        ][constant_interval]
        mutable.data["weighted_variance"][mutable_interval] += constant.data[
            "weighted_variance"
        ][constant_interval]
        mutable.data["flag"][mutable_interval] |= constant.data["flag"][constant_interval]  # type: ignore
        if correlated_variance is not None:
            mutable.data["weighted_variance"][mutable_interval] += correlated_variance

    def get_spectrum(self) -> Spectrum:
        if not self.spectrum_slices:
            return Spectrum(intensity=[], frequency=[], noise=[], flag=[])

        total_length = (
            sum(map(lambda slc: slc.stop - slc.start + 1, self.spectrum_slices)) - 1
        )
        frequency = numpy.full(total_length, numpy.nan)
        intensity = numpy.full(total_length, numpy.nan)
        noise = numpy.full(total_length, numpy.nan)
        flag = numpy.full(total_length, Spectrum.FlagReason.NAN.value)
        filled_length = 0
        for spectrum_slice in self.spectrum_slices:
            current_length = spectrum_slice.stop - spectrum_slice.start
            current_interval = slice(
                spectrum_slice.start - spectrum_slice.buffer_start,
                spectrum_slice.stop - spectrum_slice.buffer_start,
            )
            fill_interval = slice(filled_length, filled_length + current_length)
            frequency[fill_interval] = self.transformer.forward(
                numpy.arange(spectrum_slice.start, spectrum_slice.stop)
            )
            intensity[fill_interval] = (
                spectrum_slice.data["weighted_intensity"][current_interval]
                / spectrum_slice.data["weight"][current_interval]
            )
            noise[fill_interval] = (
                numpy.sqrt(spectrum_slice.data["weighted_variance"][current_interval])
                / spectrum_slice.data["weight"][current_interval]
            )
            flag[fill_interval] = spectrum_slice.data["flag"][current_interval]
            filled_length += current_length + 1
        return Spectrum(
            intensity=intensity, frequency=frequency, noise=noise, flag=flag
        )


class ExposureAggregator(Aggregator[Exposure]):

    def __init__(self, *args, **kwargs):
        super().__init__(Exposure, "frequency", *args, **kwargs)

    def _init_spectrum_slice(self, spectrum_slice: Aggregator.SpectrumSlice):
        buffer_length = spectrum_slice.buffer_stop - spectrum_slice.buffer_start
        spectrum_slice.data["exposure"] = numpy.zeros(
            buffer_length, dtype=numpy.float64
        )

    def _fill_spectrum_slice(
        self,
        spectrum_slice: Aggregator.SpectrumSlice,
        spectrum: Exposure,
        interval: slice,
        aligned_value: numpy.typing.NDArray[numpy.floating],
    ):
        nearest, out_of_bound = searchsorted_nearest(spectrum.frequency, aligned_value)
        spectrum_slice.data["exposure"][interval] = spectrum.exposure[nearest]
        spectrum_slice.data["exposure"][interval][out_of_bound] = 0.0

    def _add_spectrum_slices(
        self,
        mutable: Aggregator.SpectrumSlice,
        constant: Aggregator.SpectrumSlice,
        mutable_interval: slice,
        constant_interval: slice,
    ):
        mutable.data["exposure"][mutable_interval] += constant.data["exposure"][
            constant_interval
        ]

    def get_spectrum(self) -> Exposure:
        if not self.spectrum_slices:
            return Exposure(exposure=[], frequency=[])

        total_length = (
            sum(map(lambda slc: slc.stop - slc.start + 1, self.spectrum_slices)) - 1
        )
        frequency = numpy.full(total_length, numpy.nan)
        exposure = numpy.full(total_length, numpy.nan)
        filled_length = 0
        for spectrum_slice in self.spectrum_slices:
            current_length = spectrum_slice.stop - spectrum_slice.start
            current_interval = slice(
                spectrum_slice.start - spectrum_slice.buffer_start,
                spectrum_slice.stop - spectrum_slice.buffer_start,
            )
            fill_interval = slice(filled_length, filled_length + current_length)
            frequency[fill_interval] = self.transformer.forward(
                numpy.arange(spectrum_slice.start, spectrum_slice.stop)
            )
            exposure[fill_interval] = spectrum_slice.data["exposure"][current_interval]
            filled_length += current_length + 1
        return Exposure(exposure=exposure, frequency=frequency)
