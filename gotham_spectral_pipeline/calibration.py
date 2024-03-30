from .sdfits import HDUList, SDFits
from .spectrum import Spectrum
from .utils import datetime_parser, lru_cache
from .zenith_opacity import ZenithOpacity

import typing
import uuid

import astropy.constants  # type: ignore
import astropy.io.fits  # type: ignore
import astropy.wcs  # type: ignore
import loguru
import numpy
import numpy.typing
import pandas

__all__ = [
    "CalOnOffPairedHDUList",
    "CalOnOffPairedRows",
    "SigRefPairedHDUList",
    "SigRefPairedRows",
    "Calibration",
    "PositionSwitchedCalibration",
    "PointingCalibration",
]

CalOnOffName = typing.Literal["calon", "caloff"]
SigRefName = typing.Literal["sig", "ref"]


class CalOnOffPairedHDUList(dict[CalOnOffName, HDUList]):
    T1 = typing.TypeVar("T1")
    T2 = typing.TypeVar("T2")

    def __hash__(self):
        return hash(id(self))

    def __eq__(self, other):
        return self is other

    def get_property(
        self,
        getter: typing.Callable[[HDUList], T1],
        transformer: (
            typing.Callable[[T1], T1] | typing.Callable[[T1], T2]
        ) = lambda x: x,
        property_name: str = "Property",
    ) -> T1:
        properties = {key: getter(self[key]) for key in self.keys()}
        transformed = set(map(transformer, properties.values()))
        default_scan: typing.Final = "caloff"
        if len(transformed) > 1:
            loguru.logger.warning(
                f"{property_name} of each HDUList are not identical. Using the one of {default_scan}."
            )
        return properties[default_scan]


class CalOnOffPairedRows(dict[CalOnOffName, pandas.DataFrame]):

    def get_paired_hdu(self, sdfits: SDFits) -> CalOnOffPairedHDUList:
        return CalOnOffPairedHDUList(
            {k: sdfits.get_hdulist_from_rows(v) for k, v in self.items()}
        )


class SigRefPairedHDUList(dict[SigRefName, CalOnOffPairedHDUList]):
    pass


class SigRefPairedRows(dict[SigRefName, CalOnOffPairedRows]):
    metadata: dict[str, typing.Any]

    def __init__(self, metadata: dict[str, typing.Any] = dict(), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata = metadata

    def get_paired_hdu(self, sdfits: SDFits) -> SigRefPairedHDUList:
        return SigRefPairedHDUList(
            {k: v.get_paired_hdu(sdfits) for k, v, in self.items()}
        )


class Calibration:

    @classmethod
    def get_observed_datetime(cls, hdulist: HDUList) -> str:
        raise NotImplementedError()

    @classmethod
    def get_observed_elevation(cls, hdulist: HDUList) -> float:
        raise NotImplementedError()

    @classmethod
    def get_observed_frequency(
        cls,
        hdulist: HDUList,
        loc: typing.Literal["center", "edge"] = "center",
        unit: str = "Hz",
    ) -> numpy.typing.NDArray[numpy.floating] | None:
        raise NotImplementedError()

    @classmethod
    def get_corrected_frequency(
        cls,
        hdulist: HDUList,
        loc: typing.Literal["center", "edge"] = "center",
        unit: str = "Hz",
    ) -> numpy.typing.NDArray[numpy.floating] | None:
        raise NotImplementedError()

    @classmethod
    def get_intensity_raw_count(
        cls, hdulist: HDUList
    ) -> numpy.typing.NDArray[numpy.floating]:
        raise NotImplementedError()

    @classmethod
    def get_noise(
        cls, hdulist: HDUList, Tsys: float
    ) -> numpy.typing.NDArray[numpy.floating]:
        raise NotImplementedError()

    @classmethod
    def get_calibration_temperature(cls, hdulist: HDUList) -> float:
        raise NotImplementedError()

    @classmethod
    @lru_cache(maxsize=128)
    def get_system_temperature(
        cls,
        calonoffpair: CalOnOffPairedHDUList,
        Tcal: float | None = None,
        trim_fraction: float = 0.0,
    ) -> float:
        if Tcal is None:
            Tcal = calonoffpair.get_property(
                getter=cls.get_calibration_temperature, property_name="Tcal"
            )

        caloff = cls.get_intensity_raw_count(calonoffpair["caloff"])
        calon = cls.get_intensity_raw_count(calonoffpair["calon"])

        trim_length = int(caloff.size * trim_fraction)
        if trim_length != 0:
            caloff_trimmed = caloff[trim_length:-trim_length]
            calon_trimmed = calon[trim_length:-trim_length]
        else:
            caloff_trimmed = caloff
            calon_trimmed = calon

        Tsys = Tcal * (
            0.5 + caloff_trimmed.mean() / (calon_trimmed - caloff_trimmed).mean()
        )
        return Tsys

    @typing.overload
    @classmethod
    def get_total_power_spectrum(
        cls,
        calonoffpair: CalOnOffPairedHDUList,
        Tcal: float | None = None,
        Tsys: float | None = None,
        ref_calonoffpair: CalOnOffPairedHDUList | None = None,
        freq_kwargs: dict = dict(),
        *,
        return_metadata: typing.Literal[False] = ...,
    ) -> Spectrum | None:
        ...

    @typing.overload
    @classmethod
    def get_total_power_spectrum(
        cls,
        calonoffpair: CalOnOffPairedHDUList,
        Tcal: float | None = None,
        Tsys: float | None = None,
        ref_calonoffpair: CalOnOffPairedHDUList | None = None,
        freq_kwargs: dict = dict(),
        *,
        return_metadata: typing.Literal[True],
    ) -> tuple[Spectrum | None, dict]:
        ...

    @typing.overload
    @classmethod
    def get_total_power_spectrum(
        cls,
        calonoffpair: CalOnOffPairedHDUList,
        Tcal: float | None = None,
        Tsys: float | None = None,
        ref_calonoffpair: CalOnOffPairedHDUList | None = None,
        freq_kwargs: dict = dict(),
        *,
        return_metadata: bool = False,
    ) -> (Spectrum | None) | tuple[Spectrum | None, dict]:
        ...

    @classmethod
    def get_total_power_spectrum(
        cls,
        calonoffpair: CalOnOffPairedHDUList,
        Tcal: float | None = None,
        Tsys: float | None = None,
        ref_calonoffpair: CalOnOffPairedHDUList | None = None,
        freq_kwargs: dict = dict(),
        *,
        return_metadata: bool = False,
    ) -> (Spectrum | None) | tuple[Spectrum | None, dict]:
        frequency = calonoffpair.get_property(
            lambda hdulist: cls.get_corrected_frequency(hdulist, **freq_kwargs),
            transformer=lambda ndarray: ndarray.tobytes(),
            property_name="Corrected frequency grid",
        )
        metadata: dict[str, typing.Any] = dict()

        def with_metadata(result: Spectrum | None):
            if return_metadata:
                return result, metadata
            else:
                return result

        if frequency is None:
            return with_metadata(None)

        if Tcal is None:
            Tcal = calonoffpair.get_property(
                getter=cls.get_calibration_temperature, property_name="Tcal"
            )
        metadata["Tcal"] = Tcal

        if Tsys is None:
            Tsys = cls.get_system_temperature(
                calonoffpair, Tcal=Tcal, trim_fraction=0.1
            )
        metadata["Tsys"] = Tsys

        sig_calon_raw = cls.get_intensity_raw_count(calonoffpair["calon"])
        sig_caloff_raw = cls.get_intensity_raw_count(calonoffpair["caloff"])
        if ref_calonoffpair is None:
            countPerK = 0.5 * (sig_calon_raw + sig_caloff_raw) / Tsys
        else:
            countPerK = (
                0.5
                * (
                    cls.get_intensity_raw_count(ref_calonoffpair["calon"])
                    + cls.get_intensity_raw_count(ref_calonoffpair["caloff"])
                )
                / Tsys
            )
        noise_calon = cls.get_noise(calonoffpair["calon"], Tsys)
        noise_caloff = cls.get_noise(calonoffpair["caloff"], Tsys)
        flag = numpy.zeros_like(frequency, dtype=int)

        metadata["calon"] = sig_calon = Spectrum(
            intensity=sig_calon_raw / countPerK,
            frequency=frequency,
            noise=noise_calon,
            flag=flag,
        )
        metadata["caloff"] = sig_caloff = Spectrum(
            intensity=sig_caloff_raw / countPerK,
            frequency=frequency,
            noise=noise_caloff,
            flag=flag,
        )

        sig = 0.5 * (sig_calon + sig_caloff) - 0.5 * Tcal
        return with_metadata(sig)

    @classmethod
    def get_temperature_correction_factor(
        cls,
        hdulist: HDUList,
        zenith_opacity: ZenithOpacity,
        eta_l: float = 0.99,
    ) -> Spectrum | None:
        frequency = cls.get_observed_frequency(hdulist)
        if frequency is None:
            return None

        timestamp = datetime_parser(cls.get_observed_datetime(hdulist)).timestamp()
        tau = zenith_opacity.get_opacity(timestamp, frequency)
        elevation = cls.get_observed_elevation(hdulist)
        return Spectrum(
            intensity=numpy.exp(tau / numpy.sin(numpy.deg2rad(elevation))) / eta_l
        )


class PositionSwitchedCalibration(Calibration):

    @classmethod
    def pair_up_rows(
        cls, rows: pandas.DataFrame, **kwargs: dict[str, typing.Any]
    ) -> list[SigRefPairedRows]:
        query = [f"{key.upper()} == {val!r}" for key, val in kwargs.items()]
        rows = rows.query(" and ".join(query)) if query else rows

        position_switched_rows = rows.query("PROCEDURE in ['OffOn', 'OnOff']")
        if len(rows) != len(position_switched_rows):
            loguru.logger.warning(
                f"Found {len(rows) - len(position_switched_rows)} rows using procedure that is not supported. Supported procedures are ['OffOn', 'OnOff']."
            )

        rows = position_switched_rows
        is_first_scan = (rows.PROCEDURE == "OffOn") == (rows.PROCSCAN == "OFF")
        rows["PAIRED_OFFSCAN"] = numpy.where(is_first_scan, rows.SCAN, rows.SCAN - 1)
        groups = rows.groupby(["SOURCE", "PAIRED_OFFSCAN", "SAMPLER"])
        paired_up_rows: list[SigRefPairedRows] = list()
        for group, rows_in_group in groups:
            ref_caloff = rows_in_group.query("PROCSCAN == 'OFF' and CAL == 'F'")
            ref_calon = rows_in_group.query("PROCSCAN == 'OFF' and CAL == 'T'")
            sig_caloff = rows_in_group.query("PROCSCAN == 'ON' and CAL == 'F'")
            sig_calon = rows_in_group.query("PROCSCAN == 'ON' and CAL == 'T'")
            if not (
                len(ref_caloff) == len(ref_calon) == len(sig_caloff) == len(sig_calon)
            ):
                loguru.logger.warning(
                    f"Numbers of rows in each scan do not match for {group = }. {len(ref_caloff) = }, {len(ref_calon) = }, {len(sig_caloff) = }, {len(sig_calon) = }."
                )
                continue
            paired_up_rows.extend(
                [
                    SigRefPairedRows(
                        ref=CalOnOffPairedRows(
                            caloff=pandas.DataFrame([ref_caloff_]),
                            calon=pandas.DataFrame([ref_calon_]),
                        ),
                        sig=CalOnOffPairedRows(
                            caloff=pandas.DataFrame([sig_caloff_]),
                            calon=pandas.DataFrame([sig_calon_]),
                        ),
                        metadata=dict(
                            group=dict(
                                source=group[0], offscan=group[1], sampler=group[2]
                            )
                        ),
                    )
                    for (
                        (_, ref_caloff_),
                        (_, ref_calon_),
                        (_, sig_caloff_),
                        (_, sig_calon_),
                    ) in zip(
                        ref_caloff.iterrows(),
                        ref_calon.iterrows(),
                        sig_caloff.iterrows(),
                        sig_calon.iterrows(),
                    )
                ]
            )
        return paired_up_rows

    @classmethod
    def get_observed_datetime(cls, hdulist: HDUList) -> str:
        return hdulist[0].header["DATE-OBS"]

    @classmethod
    def get_observed_elevation(cls, hdulist: HDUList) -> float:
        return hdulist[0].header["ELEVATIO"]

    @classmethod
    def get_observed_frequency(
        cls,
        hdulist: HDUList,
        loc: typing.Literal["center", "edge"] = "center",
        unit: str = "Hz",
    ) -> numpy.typing.NDArray[numpy.floating] | None:
        wcs = astropy.wcs.WCS(hdulist[0]).spectral
        if wcs.naxis != 1:
            loguru.logger.error(f"Expecting one spectral axis. Found {wcs.naxis}.")
            return None

        if loc == "center":
            return wcs.pixel_to_world(numpy.arange(wcs.pixel_shape[0])).to_value(unit)
        if loc == "edge":
            return wcs.pixel_to_world(
                numpy.arange(wcs.pixel_shape[0] + 1) - 0.5
            ).to_value(unit)
        loguru.logger.error("Invalid location. Supported are ['center', 'edge']")
        return None

    @classmethod
    def get_corrected_frequency(
        cls,
        hdulist: HDUList,
        loc: typing.Literal["center", "edge"] = "center",
        unit: str = "Hz",
        method: typing.Literal["default", "four_chunks"] = "default",
    ) -> numpy.typing.NDArray[numpy.floating] | None:
        frequency = cls.get_observed_frequency(hdulist, loc=loc, unit=unit)
        if frequency is None:
            return None

        vframe = hdulist[0].header["VFRAME"]
        beta = vframe / astropy.constants.c.to_value("m/s")
        if method == "default":
            doppler = numpy.sqrt((1 + beta) / (1 - beta))
            corrected_frequency = frequency * doppler
            return corrected_frequency
        if method == "four_chunks":
            corrected_frequency = frequency.copy()
            for chunk in numpy.array_split(corrected_frequency, 4):
                central_frequency = 0.5 * (chunk[0] + chunk[-1])
                chunk += central_frequency * beta
            return corrected_frequency
        loguru.logger.error("Invalid method. Supported are ['default', 'four_chunks']")
        return None

    @classmethod
    @lru_cache(maxsize=128)
    def get_intensity_raw_count(
        cls, hdulist: HDUList
    ) -> numpy.typing.NDArray[numpy.floating]:
        return hdulist[0].data.squeeze()

    @classmethod
    def get_noise(
        cls, hdulist: HDUList, Tsys: float
    ) -> numpy.typing.NDArray[numpy.floating]:
        frequency_resolution = hdulist[0].header["FREQRES"]
        exposure = hdulist[0].header["EXPOSURE"]
        return numpy.full_like(
            hdulist[0].data.squeeze(),
            Tsys / numpy.sqrt(frequency_resolution * exposure),
        )

    @classmethod
    def get_calibration_temperature(cls, hdulist: HDUList) -> float:
        return hdulist[0].header["TCAL"]

    @classmethod
    def should_be_discarded(
        cls, sigrefpair: SigRefPairedHDUList, threshold=0.10
    ) -> bool:
        if any(
            hdulist[0].header["EXPOSURE"] == 0.0
            for calonoffpair in sigrefpair.values()
            for hdulist in calonoffpair.values()
        ):
            return True

        if numpy.all(
            cls.get_intensity_raw_count(sigrefpair["ref"]["caloff"])
            == cls.get_intensity_raw_count(sigrefpair["sig"]["caloff"])
        ) and numpy.all(
            cls.get_intensity_raw_count(sigrefpair["ref"]["calon"])
            == cls.get_intensity_raw_count(sigrefpair["sig"]["calon"])
        ):
            return True

        averages = {
            sigref: {
                calonoff: numpy.nanmean(cls.get_intensity_raw_count(hdulist))
                for calonoff, hdulist in calonoffpair.items()
            }
            for sigref, calonoffpair in sigrefpair.items()
        }
        if any(
            numpy.isnan(avg)
            for calonoffavgs in averages.values()
            for avg in calonoffavgs.values()
        ):
            return True

        percentage_difference = [
            abs(a - b) / (0.5 * (a + b))
            for a, b in [
                (averages["ref"]["caloff"], averages["sig"]["caloff"]),
                (averages["ref"]["calon"], averages["sig"]["calon"]),
            ]
        ]
        if any(
            not numpy.isfinite(pd) or pd > threshold for pd in percentage_difference
        ):
            return True

        return False

    @typing.overload
    @classmethod
    def get_calibrated_spectrum(
        cls,
        sigrefpair: SigRefPairedHDUList,
        freq_kwargs: dict = dict(),
        *,
        return_metadata: typing.Literal[False] = ...,
    ) -> Spectrum | None:
        ...

    @typing.overload
    @classmethod
    def get_calibrated_spectrum(
        cls,
        sigrefpair: SigRefPairedHDUList,
        freq_kwargs: dict = dict(),
        *,
        return_metadata: typing.Literal[True],
    ) -> tuple[Spectrum | None, dict]:
        ...

    @typing.overload
    @classmethod
    def get_calibrated_spectrum(
        cls,
        sigrefpair: SigRefPairedHDUList,
        freq_kwargs: dict = dict(),
        *,
        return_metadata: bool = False,
    ) -> (Spectrum | None) | tuple[Spectrum | None, dict]:
        ...

    @classmethod
    def get_calibrated_spectrum(
        cls,
        sigrefpair: SigRefPairedHDUList,
        freq_kwargs: dict = dict(),
        *,
        return_metadata: bool = False,
    ) -> (Spectrum | None) | tuple[Spectrum | None, dict]:
        metadata: dict[str, typing.Any] = dict()

        def with_metadata(result: Spectrum | None):
            if return_metadata:
                return result, metadata
            else:
                return result

        sig_calonoffpair = sigrefpair["sig"]
        ref_calonoffpair = sigrefpair["ref"]
        Tcal = ref_calonoffpair.get_property(
            cls.get_calibration_temperature, property_name="Tcal"
        )
        Tsys = cls.get_system_temperature(
            ref_calonoffpair, Tcal=Tcal, trim_fraction=0.1
        )
        metadata["Tcal"] = Tcal
        metadata["Tsys"] = Tsys

        ref_total_power, ref_metadata = cls.get_total_power_spectrum(
            ref_calonoffpair,
            Tcal=Tcal,
            Tsys=Tsys,
            ref_calonoffpair=ref_calonoffpair,
            freq_kwargs=freq_kwargs,
            return_metadata=True,
        )
        if ref_total_power is None:
            return with_metadata(None)
        metadata["ref_total_power"] = ref_total_power
        metadata["ref_calon"] = ref_metadata["calon"]
        metadata["ref_caloff"] = ref_metadata["caloff"]

        sig_total_power, sig_metadata = cls.get_total_power_spectrum(
            sig_calonoffpair,
            Tcal=Tcal,
            Tsys=Tsys,
            ref_calonoffpair=ref_calonoffpair,
            freq_kwargs=freq_kwargs,
            return_metadata=True,
        )
        if sig_total_power is None:
            return with_metadata(None)
        metadata["sig_total_power"] = sig_total_power
        metadata["sig_calon"] = sig_metadata["calon"]
        metadata["sig_caloff"] = sig_metadata["caloff"]

        return with_metadata(sig_total_power - ref_total_power)


class PointingCalibration(Calibration):

    @classmethod
    def pair_up_rows(
        cls, rows: pandas.DataFrame, **kwargs: dict[str, typing.Any]
    ) -> list[SigRefPairedRows]:
        query = [f"{key.upper()} == {val!r}" for key, val in kwargs.items()]
        rows = rows.query(" and ".join(query)) if query else rows

        pointing_rows = rows.query("PROCTYPE.str.strip() == 'POINTING'")
        if len(rows) != len(pointing_rows):
            loguru.logger.warning(
                f"Found {len(rows) - len(pointing_rows)} rows that are not pointing scans."
            )

        rows = pointing_rows
        groups = rows.groupby(["OBJECT", "SCAN", "PLNUM"])
        paired_up_rows: list[SigRefPairedRows] = list()
        for group, rows_in_group in groups:
            ref_caloff = rows_in_group.query("FDNUM == 1 and CAL == 'F'")
            ref_calon = rows_in_group.query("FDNUM == 1 and CAL == 'T'")
            sig_caloff = rows_in_group.query("FDNUM == 0 and CAL == 'F'")
            sig_calon = rows_in_group.query("FDNUM == 0 and CAL == 'T'")
            if not (
                len(ref_caloff) == len(ref_calon) == len(sig_caloff) == len(sig_calon)
            ):
                loguru.logger.warning(
                    f"Numbers of rows in each scan do not match for {group = }. {len(ref_caloff) = }, {len(ref_calon) = }, {len(sig_caloff) = }, {len(sig_calon) = }."
                )
                continue
            paired_up_rows.append(
                SigRefPairedRows(
                    ref=CalOnOffPairedRows(caloff=ref_caloff, calon=ref_calon),
                    sig=CalOnOffPairedRows(caloff=sig_caloff, calon=sig_calon),
                    metadata=dict(
                        group=dict(object=group[0], scan=group[1], plnum=group[2])
                    ),
                )
            )
        return paired_up_rows
