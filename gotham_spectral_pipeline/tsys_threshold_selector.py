import typing

__all__ = [
    "TsysLookupTable",
    "GbtTsysLookupTable",
]


class TsysThresholdSelector:

    def get_threshold(self, frequency: float) -> float | None:
        raise NotImplementedError


class TsysLookupTable(TsysThresholdSelector):
    # Lookup table is a list of 3-tuple where the elements are
    # minimum frequency, maximum frequency, and system temperature.
    lookup_table: list[tuple[float, float, float]]
    default: float | str | None
    reduce: typing.Callable[[list[float]], float] | None

    def __init__(
        self,
        lookup_table: list[tuple[float, float, float]],
        *,
        default: float | str | None = None,
        reduce: typing.Callable[[list[float]], float] | None = None,
    ):
        self.lookup_table = lookup_table
        self.default = default
        self.reduce = reduce

    def get_threshold(self, frequency: float) -> float | None:
        values = [
            value
            for min_freq, max_freq, value in self.lookup_table
            if min_freq <= frequency <= max_freq
        ]
        if not values:
            if not isinstance(self.default, str):
                return self.default
            if self.default == "closest":
                values_with_distance = [
                    (value, min(abs(frequency - min_freq), abs(frequency - max_freq)))
                    for min_freq, max_freq, value in self.lookup_table
                ]
                return min(values_with_distance, key=lambda x: x[1])[0]
            raise RuntimeError(f"default = {self.default} is not supported.")
        if len(values) == 1:
            return values[0]
        if self.reduce is None:
            raise RuntimeError(
                "More than one lookup values but reduce function is not defined."
            )
        return self.reduce(values)


class GbtTsysLookupTable(TsysLookupTable):
    # Lookup values taken from Table 2.2 of the
    # "Observing With The Green Bank Telescope" handbook:
    # https://www.gb.nrao.edu/scienceDocs/GBTog.pdf

    def __init__(
        self,
        *,
        default: float | str | None = "closest",
        reduce: typing.Callable[[list[float]], float] | None = max,
    ):
        super().__init__(
            [
                (1.15e9, 1.73e9, 20),
                (1.73e9, 2.60e9, 22),
                (3.95e9, 7.80e9, 18),
                (8.00e9, 10.1e9, 27),
                (12.0e9, 15.4e9, 30),
                (18.0e9, 26.5e9, 45),
                (26.0e9, 31.0e9, 35),
                (30.5e9, 37.0e9, 30),
                (36.0e9, 39.5e9, 45),
                (38.2e9, 49.8e9, 134),
                (67.0e9, 74.0e9, 160),
                (73.0e9, 80.0e9, 120),
                (79.0e9, 86.0e9, 100),
                (85.0e9, 92.0e9, 110),
                (74.0e9, 116.0e9, 110),
            ],
            default=default,
            reduce=reduce,
        )
