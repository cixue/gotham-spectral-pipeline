import numpy
import numpy.typing


class Spectrum:
    frequency: numpy.typing.NDArray[numpy.floating]
    intensity: numpy.typing.NDArray[numpy.floating]
    noise: numpy.typing.NDArray[numpy.floating]

    def __init__(
        self,
        frequency: numpy.typing.ArrayLike,
        intensity: numpy.typing.ArrayLike,
        noise: numpy.typing.ArrayLike,
    ):
        self.frequency = numpy.array(frequency)
        self.intensity = numpy.array(intensity)
        self.noise = numpy.array(noise)
