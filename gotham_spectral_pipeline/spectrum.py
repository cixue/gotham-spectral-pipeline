import enum

import loguru
import numpy
import numpy.typing
import scipy.special


class Spectrum:
    frequency: numpy.typing.NDArray[numpy.floating]
    intensity: numpy.typing.NDArray[numpy.floating]
    noise: numpy.typing.NDArray[numpy.floating]
    flag: numpy.typing.NDArray[numpy.object_]

    class FlagReason(enum.Enum):
        NOT_FLAGGED = 0
        CHUNK_EDGES = 1
        RFI = 2

    def __init__(
        self,
        frequency: numpy.typing.ArrayLike,
        intensity: numpy.typing.ArrayLike,
        noise: numpy.typing.ArrayLike,
    ):
        self.frequency = numpy.array(frequency)
        self.intensity = numpy.array(intensity)
        self.noise = numpy.array(noise)
        if not (self.frequency.shape == self.intensity.shape == self.noise.shape):
            loguru.logger.error(
                "Frequency, intensity and noise do not have the same shape."
            )
            self.frequency = self.intensity = self.noise = numpy.empty(0)
            return

        self.flag = numpy.full_like(
            self.frequency, Spectrum.FlagReason.NOT_FLAGGED, dtype=object
        )

    @property
    def flagged(self) -> numpy.typing.NDArray[numpy.bool_]:
        return self.flag != Spectrum.FlagReason.NOT_FLAGGED

    def flag_rfi(self, nadjacent: int = 3, confidence: float = 0.999999):
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
        is_rfi = numpy.where(scipy.special.chdtr(width, chi_squared) > confidence)[0]
        for i in range(width):
            self.flag[i : delta.size + i][is_rfi] = Spectrum.FlagReason.RFI

    def flag_head_tail(
        self, *, fraction: float | None = None, nchannel: int | None = None
    ):
        if nchannel is None:
            if fraction is None:
                loguru.logger.error(
                    "Either but not both fraction or nchannel should be set."
                )
                return
            nchannel = int(self.frequency.size * fraction)

        self.flag[:nchannel] = self.flag[-nchannel:] = Spectrum.FlagReason.CHUNK_EDGES
