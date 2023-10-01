import enum
import functools
import typing

import astropy.timeseries
import loguru
import numpy
import numpy.polynomial.polynomial as poly
import numpy.typing
import scipy.special


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

    def fit_polynomial_baseline(
        self, degree: int, mask: numpy.typing.NDArray[numpy.bool_] | None = None
    ) -> typing.Callable[
        [numpy.typing.NDArray[numpy.floating]], numpy.typing.NDArray[numpy.floating]
    ]:
        if mask is None:
            mask = self.flagged
        else:
            mask = mask | self.flagged

        frequency = self.frequency[~mask]
        intensity = self.intensity[~mask]
        if self.noise is None:
            polynomial = poly.Polynomial.fit(frequency, intensity, degree)
        else:
            noise = self.noise[~mask]
            polynomial = poly.Polynomial.fit(frequency, intensity, degree, w=1 / noise)
        return polynomial

    def fit_lomb_scargle_baseline(
        self,
        max_cycle: float = 32.0,
        mask: numpy.typing.NDArray[numpy.bool_] | None = None,
    ) -> typing.Callable[
        [numpy.typing.NDArray[numpy.floating]], numpy.typing.NDArray[numpy.floating]
    ]:
        if mask is None:
            mask = self.flagged
        else:
            mask = mask | self.flagged

        frequency = self.frequency[~mask]
        intensity = self.intensity[~mask]
        if self.noise is None:
            lomb_scargle = astropy.timeseries.LombScargle(frequency, intensity)
        else:
            noise = self.noise[~mask]
            lomb_scargle = astropy.timeseries.LombScargle(frequency, intensity, noise)

        min_frequency = (frequency.max() - frequency.min()) / max_cycle
        max_ls_frequency = 1 / min_frequency
        ls_frequency, power = lomb_scargle.autopower(
            method="fast", maximum_frequency=max_ls_frequency
        )
        best_ls_frequency = ls_frequency[numpy.argmax(power)]
        return functools.partial(lomb_scargle.model, frequency=best_ls_frequency)
