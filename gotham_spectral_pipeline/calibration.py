from .sdfits import SDFits
from .spectrum import Spectrum
from .utils import datetime_parser
from .zenith_opacity import ZenithOpacity

import functools
import typing

import astropy.constants  # type: ignore
import astropy.io.fits  # type: ignore
import astropy.wcs  # type: ignore
import loguru
import numpy
import numpy.typing
import pandas

__all__ = [
    "CalOnOffPairedHDU",
    "CalOnOffPairedRow",
    "SigRefPairedHDU",
    "SigRefPairedRow",
    "Calibration",
    "PositionSwitchedCalibration",
    "PointingCalibration",
]

CalOnOffName = typing.Literal["calon", "caloff"]
SigRefName = typing.Literal["sig", "ref"]


class CalOnOffPairedHDU(dict[CalOnOffName, astropy.io.fits.PrimaryHDU]):
    T1 = typing.TypeVar("T1")
    T2 = typing.TypeVar("T2")

    def __hash__(self):
        return hash(tuple(sorted(self.items())))

    def get_property(
        self,
        getter: typing.Callable[[astropy.io.fits.PrimaryHDU], T1],
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
                f"{property_name} of each HDU are not identical. Using the one of {default_scan}."
            )
        return properties[default_scan]


class CalOnOffPairedRow(dict[CalOnOffName, pandas.Series]):

    def get_paired_hdu(self, sdfits: SDFits) -> CalOnOffPairedHDU:
        return CalOnOffPairedHDU(
            {k: sdfits.get_hdu_from_row(v) for k, v in self.items()}
        )


class SigRefPairedHDU(dict[SigRefName, CalOnOffPairedHDU]):
    pass


class SigRefPairedRow(dict[SigRefName, CalOnOffPairedRow]):

    def get_paired_hdu(self, sdfits: SDFits) -> SigRefPairedHDU:
        return SigRefPairedHDU({k: v.get_paired_hdu(sdfits) for k, v, in self.items()})


class Calibration:

    @classmethod
    def get_observed_frequency(
        cls,
        hdu: astropy.io.fits.PrimaryHDU,
        loc: typing.Literal["center", "edge"] = "center",
        unit: str = "Hz",
    ) -> numpy.typing.NDArray[numpy.floating] | None:
        wcs = astropy.wcs.WCS(hdu).spectral
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
        hdu: astropy.io.fits.PrimaryHDU,
        loc: typing.Literal["center", "edge"] = "center",
        unit: str = "Hz",
        method: typing.Literal["default", "four_chunks"] = "default",
    ) -> numpy.typing.NDArray[numpy.floating] | None:
        frequency = cls.get_observed_frequency(hdu, loc=loc, unit=unit)
        if frequency is None:
            return None

        vframe = hdu.header["VFRAME"]
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
    def get_calibration_temperature(
        cls,
        calonoffpair: CalOnOffPairedHDU,
    ) -> float:
        Tcal = calonoffpair.get_property(
            getter=lambda hdu: hdu.header["TCAL"],
            property_name="TCAL",
        )
        return Tcal

    @functools.lru_cache(maxsize=4)
    @classmethod
    def get_system_temperature(
        cls,
        calonoffpair: CalOnOffPairedHDU,
        Tcal: float | None = None,
        trim_fraction: float = 0.1,
    ) -> float:
        if Tcal is None:
            Tcal = cls.get_calibration_temperature(calonoffpair)

        caloff = calonoffpair["caloff"].data.squeeze()
        calon = calonoffpair["calon"].data.squeeze()

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

    @classmethod
    def get_total_power_spectrum(
        cls,
        calonoffpair: CalOnOffPairedHDU,
        Tcal: float | None = None,
        Tsys: float | None = None,
        ref_calonoffpair: CalOnOffPairedHDU | None = None,
        freq_kwargs: dict = dict(),
    ) -> Spectrum | None:
        frequency = calonoffpair.get_property(
            lambda hdu: cls.get_corrected_frequency(hdu, **freq_kwargs),
            transformer=lambda ndarray: ndarray.tobytes(),
            property_name="Frequency grid",
        )
        if frequency is None:
            return None

        if Tcal is None:
            Tcal = cls.get_calibration_temperature(calonoffpair)

        if Tsys is None:
            Tsys = cls.get_system_temperature(calonoffpair, Tcal=Tcal)

        sig = 0.5 * (
            calonoffpair["calon"].data.squeeze() + calonoffpair["caloff"].data.squeeze()
        )
        if ref_calonoffpair is None:
            ref = sig
        else:
            ref = 0.5 * (
                ref_calonoffpair["calon"].data.squeeze()
                + ref_calonoffpair["caloff"].data.squeeze()
            )
        intensity = Tsys * sig / ref - 0.5 * Tcal

        frequency_resolution = calonoffpair.get_property(
            lambda hdu: hdu.header["FREQRES"],
            property_name="FREQRES",
        )
        exposure = (
            calonoffpair["calon"].header["EXPOSURE"]
            + calonoffpair["caloff"].header["EXPOSURE"]
        )
        noise = numpy.full_like(
            frequency, Tsys / numpy.sqrt(frequency_resolution * exposure)
        )

        return Spectrum(frequency=frequency, intensity=intensity, noise=noise)

    @classmethod
    def get_temperature_correction_factor(
        cls,
        calonoffpair: CalOnOffPairedHDU,
        zenith_opacity: ZenithOpacity,
        eta_l: float = 0.99,
    ) -> numpy.typing.NDArray[numpy.floating] | None:
        frequency = cls.get_observed_frequency(
            calonoffpair, loc="center", unit="Hz"
        )
        if frequency is None:
            return None

        timestamp = datetime_parser(
            calonoffpair["caloff"].header["DATE-OBS"]
        ).timestamp()
        tau = zenith_opacity.get_opacity(timestamp, frequency)
        elevation = calonoffpair["caloff"].header["ELEVATIO"]
        return numpy.exp(tau / numpy.sin(numpy.deg2rad(elevation))) / eta_l


class PositionSwitchedCalibration(Calibration):

    @classmethod
    def pair_up_rows(
        cls, rows: pandas.DataFrame, **kwargs: dict[str, typing.Any]
    ) -> list[list[SigRefPairedRow]]:
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
        paired_up_rows: list[list[SigRefPairedRow]] = list()
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
            paired_up_rows.append(
                [
                    SigRefPairedRow(
                        ref=CalOnOffPairedRow(caloff=ref_caloff_, calon=ref_calon_),
                        sig=CalOnOffPairedRow(caloff=sig_caloff_, calon=sig_calon_),
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
    def should_be_discarded(cls, sigrefpair: SigRefPairedHDU, threshold=0.10) -> bool:
        if any(
            hdu.header["EXPOSURE"] == 0.0
            for calonoffpair in sigrefpair.values()
            for hdu in calonoffpair.values()
        ):
            return True

        if numpy.all(
            sigrefpair["ref"]["caloff"].data == sigrefpair["sig"]["caloff"].data
        ) and numpy.all(
            sigrefpair["ref"]["calon"].data == sigrefpair["sig"]["calon"].data
        ):
            return True

        averages = {
            sigref: {
                calonoff: numpy.nanmean(hdu.data)
                for calonoff, hdu in calonoffpair.items()
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

    @classmethod
    def get_calibrated_spectrum(
        cls,
        sigrefpair: SigRefPairedHDU,
        freq_kwargs: dict = dict(),
    ) -> Spectrum | None:
        if cls.should_be_discarded(sigrefpair):
            return None

        sig_calonoffpair = sigrefpair["sig"]
        ref_calonoffpair = sigrefpair["ref"]
        Tcal = cls.get_calibration_temperature(ref_calonoffpair)
        Tsys = cls.get_system_temperature(ref_calonoffpair, Tcal=Tcal)

        ref_total_power = cls.get_total_power_spectrum(
            ref_calonoffpair,
            Tcal=Tcal,
            Tsys=Tsys,
            ref_calonoffpair=ref_calonoffpair,
            freq_kwargs=freq_kwargs,
        )
        if ref_total_power is None:
            return None

        sig_total_power = cls.get_total_power_spectrum(
            sig_calonoffpair,
            Tcal=Tcal,
            Tsys=Tsys,
            ref_calonoffpair=ref_calonoffpair,
            freq_kwargs=freq_kwargs,
        )
        if sig_total_power is None:
            return None

        return sig_total_power - ref_total_power


class PointingCalibration(Calibration):

    @classmethod
    def pair_up_rows(
        cls, rows: pandas.DataFrame, **kwargs: dict[str, typing.Any]
    ) -> list[list[SigRefPairedRow]]:
        query = [f"{key.upper()} == {val!r}" for key, val in kwargs.items()]
        rows = rows.query(" and ".join(query)) if query else rows

        pointing_rows = rows.query("PROCTYPE.str.strip() == 'POINTING'")
        if len(rows) != len(pointing_rows):
            loguru.logger.warning(
                f"Found {len(rows) - len(pointing_rows)} rows that are not pointing scans."
            )

        rows = pointing_rows
        groups = rows.groupby(["OBJECT", "SCAN", "PLNUM"])
        paired_up_rows: list[list[SigRefPairedRow]] = list()
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
                [
                    SigRefPairedRow(
                        ref=CalOnOffPairedRow(caloff=ref_caloff_, calon=ref_calon_),
                        sig=CalOnOffPairedRow(caloff=sig_caloff_, calon=sig_calon_),
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
