import astropy.constants
import numpy
import numpy.typing

__all__ = [
    "BeamEfficiency",
]


class BeamEfficiency:

    def __init__(self, mode: str = "default"):
        self.mode = mode

    def get_beam_efficiency(
        self, timestamp: float, frequency: numpy.typing.ArrayLike
    ) -> numpy.typing.NDArray[numpy.floating] | None:
        if self.mode != "default":
            return None
        epsilon = 290e-6  # surface error = 290 micron = 290 * 10**-6 m
        c = astropy.constants.c.to_value("m/s")
        coefficient = 4 * numpy.pi * epsilon / c
        return (1.37 * 0.71) * numpy.exp(-((coefficient * frequency) ** 2))
