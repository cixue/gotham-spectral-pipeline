import typing

import numpy
from numpy.polynomial.polynomial import Polynomial

__all__ = [
    "TsysThresholdSelector",
    "TsysLookupTable",
    "GbtTsysLookupTable",
    "GbtTsysHybridSelector",
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


class GbtTsysHybridSelector(TsysThresholdSelector):
    # GbtTsysLookupTable but override by a polynomial from 26 GHz to 37 GHz
    gbt_tsys_lookup_table: GbtTsysLookupTable
    polynomial: Polynomial

    def __init__(
        self,
        *,
        default: float | str | None = "closest",
        reduce: typing.Callable[[list[float]], float] | None = max,
    ):
        self.gbt_tsys_lookup_table = GbtTsysLookupTable(default=default, reduce=reduce)
        self.polynomial = Polynomial([
            -2.378406601318105459e03,
            5.279178247736501362e03,
            7.478572908927011440e02,
            -3.121567357845431616e03,
            -3.637783068873653065e03,
            -1.782972555531319358e03,
            7.575439126596024835e02,
            2.897169256656845846e03,
            3.382262301008619488e03,
            2.471259352217201467e03,
            1.314052234529612633e03,
            -5.627156568621901442e02,
            -2.214715856866484955e03,
            -2.914127280118264935e03,
            -2.909378673710764360e03,
            -2.718949545240617681e03,
            -1.316429914095534969e03,
            1.810437399686830986e02,
            1.805929383912865660e03,
            2.488094431760332100e03,
            3.054245452856589054e03,
            3.167709535433923975e03,
            2.552680117384816640e03,
            6.526384710045764450e02,
            -1.189918111895003449e03,
            -3.233092765395345850e03,
            -4.076000094738725693e03,
            -3.579067405409179173e03,
            -6.423629993712444275e02,
            5.531120535370142534e03,
        ])

    def get_threshold(self, frequency: float) -> float | None:
        if 26e9 < frequency < 37e9:
            return numpy.exp(self.polynomial(frequency / 36.3206736e9))
        return self.gbt_tsys_lookup_table.get_threshold(frequency)
