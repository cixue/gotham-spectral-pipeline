from .zenith_opacity import ZenithOpacity

import datetime
import functools
import typing

import astropy.constants
import astropy.io.fits
import astropy.wcs
import loguru
import numpy
import numpy.typing
import pandas

T1 = typing.TypeVar("T1")
T2 = typing.TypeVar("T2")
PairedScanName = typing.Literal["ref_caloff", "ref_calon", "sig_caloff", "sig_calon"]


class PairedHDU(dict[PairedScanName, astropy.io.fits.PrimaryHDU]):

    def __hash__(self):
        return hash(tuple(sorted(self.items())))

    def get_property(
        self,
        getter: typing.Callable[[astropy.io.fits.PrimaryHDU], T1],
        transformer: (
            typing.Callable[[T1], T1] | typing.Callable[[T1], T2]
        ) = lambda x: x,
        default_scan: PairedScanName = "sig_caloff",
        property_name: str = "Property",
    ) -> T1:
        properties = {key: getter(hdu) for key, hdu in self.items()}
        transformed = set(map(transformer, properties.values()))
        if len(transformed) > 1:
            loguru.logger.warning(
                f"{property_name} of each HDU are not identical. Using the one of {default_scan}."
            )
        return properties[default_scan]


class Calibration:

    @staticmethod
    def _verify_paired_hdu(paired_hdu: PairedHDU) -> bool:
        ref_caloff: numpy.typing.NDArray[numpy.floating] = paired_hdu[
            "ref_caloff"
        ].data.squeeze()
        ref_calon: numpy.typing.NDArray[numpy.floating] = paired_hdu[
            "ref_calon"
        ].data.squeeze()
        sig_caloff: numpy.typing.NDArray[numpy.floating] = paired_hdu[
            "sig_caloff"
        ].data.squeeze()
        sig_calon: numpy.typing.NDArray[numpy.floating] = paired_hdu[
            "sig_calon"
        ].data.squeeze()
        if not (
            ref_caloff.shape == ref_calon.shape == sig_caloff.shape == sig_calon.shape
        ):
            loguru.logger.error("Length of paired HDUs are not identical.")
            return False

        if ref_caloff.ndim != 1:
            loguru.logger.error("Expecting paired HDUs to be 1D.")
            return False

        return True

    @functools.lru_cache(maxsize=4)
    @staticmethod
    def get_frequency(
        paired_hdu: PairedHDU,
        loc: typing.Literal["center", "edge"] = "center",
        unit: str = "Hz",
    ) -> numpy.typing.NDArray[numpy.floating] | None:
        if not Calibration._verify_paired_hdu(paired_hdu):
            return None

        wcs = paired_hdu.get_property(
            getter=lambda hdu: astropy.wcs.WCS(hdu).spectral, transformer=str
        )
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

    @functools.lru_cache(maxsize=4)
    @staticmethod
    def get_corrected_frequency(
        paired_hdu: PairedHDU,
        loc: typing.Literal["center", "edge"] = "center",
        unit: str = "Hz",
    ) -> numpy.typing.NDArray[numpy.floating] | None:
        frequency = Calibration.get_frequency(paired_hdu, loc=loc, unit=unit)
        if frequency is None:
            return None

        vframe = paired_hdu.get_property(getter=lambda hdu: hdu.header["VFRAME"])
        beta = vframe / astropy.constants.c.to_value("m/s")
        doppler = numpy.sqrt((1 + beta) / (1 - beta))
        corrected_frequency = frequency * doppler
        return corrected_frequency

    @functools.lru_cache(maxsize=4)
    @staticmethod
    def get_system_temperature(paired_hdu: PairedHDU) -> float | None:
        if not Calibration._verify_paired_hdu(paired_hdu):
            return None

        Tcal = paired_hdu.get_property(getter=lambda hdu: hdu.header["TCAL"])

        ref_caloff: numpy.typing.NDArray[numpy.floating] = paired_hdu[
            "ref_caloff"
        ].data.squeeze()
        ref_calon: numpy.typing.NDArray[numpy.floating] = paired_hdu[
            "ref_calon"
        ].data.squeeze()

        trim_length = ref_caloff.size // 10
        ref80_caloff = ref_caloff[trim_length:-trim_length]
        ref80_calon = ref_calon[trim_length:-trim_length]

        Tsys = Tcal * (0.5 + ref80_caloff.mean() / (ref80_calon - ref80_caloff).mean())
        return Tsys

    @functools.lru_cache(maxsize=4)
    @staticmethod
    def get_antenna_temperature(
        paired_hdu: PairedHDU,
    ) -> numpy.typing.NDArray[numpy.floating] | None:
        Tsys = Calibration.get_system_temperature(paired_hdu)
        if Tsys is None:
            return None

        ref = 0.5 * (
            paired_hdu["ref_caloff"].data.squeeze()
            + paired_hdu["ref_calon"].data.squeeze()
        )
        sig = 0.5 * (
            paired_hdu["sig_caloff"].data.squeeze()
            + paired_hdu["sig_calon"].data.squeeze()
        )

        Ta = Tsys * (sig - ref) / ref
        return Ta

    @functools.lru_cache(maxsize=4)
    @staticmethod
    def get_corrected_antenna_temperature(
        paired_hdu: PairedHDU,
        zenith_opacity: ZenithOpacity,
        eta_l: float = 0.99,
    ) -> numpy.typing.NDArray[numpy.floating] | None:
        frequency = Calibration.get_frequency(paired_hdu, loc="center", unit="Hz")
        if frequency is None:
            return None

        timestamp = (
            datetime.datetime.strptime(
                paired_hdu["sig_caloff"].header["DATE-OBS"], "%Y-%m-%dT%H:%M:%S.%f"
            )
            .replace(tzinfo=datetime.timezone.utc)
            .timestamp()
        )
        tau = zenith_opacity.get_opacity(timestamp, frequency)
        elevation = paired_hdu["sig_caloff"].header["ELEVATIO"]
        Ta = Calibration.get_antenna_temperature(paired_hdu)
        Ta_corrected = Ta * numpy.exp(tau / numpy.sin(numpy.deg2rad(elevation))) / eta_l
        return Ta_corrected

    @functools.lru_cache(maxsize=4)
    @staticmethod
    def get_estimated_noise(paired_hdu: PairedHDU) -> float:
        Tsys = Calibration.get_system_temperature(paired_hdu)
        frequency_resolution = paired_hdu.get_property(
            lambda hdu: hdu.header["FREQRES"]
        )
        exposure = paired_hdu.get_property(lambda hdu: hdu.header["EXPOSURE"])
        estimated_noise = Tsys / numpy.sqrt(frequency_resolution * exposure)
        return estimated_noise

    @functools.lru_cache(maxsize=4)
    @staticmethod
    def get_corrected_estimated_noise(
        paired_hdu: PairedHDU,
        zenith_opacity: ZenithOpacity,
        eta_l: float = 0.99,
    ) -> float | None:
        frequency = Calibration.get_frequency(paired_hdu, loc="center", unit="Hz")
        if frequency is None:
            return None

        Tsys = Calibration.get_system_temperature(paired_hdu)
        frequency_resolution = paired_hdu.get_property(
            lambda hdu: hdu.header["FREQRES"]
        )
        exposure = paired_hdu.get_property(lambda hdu: hdu.header["EXPOSURE"])
        estimated_noise = Tsys / numpy.sqrt(frequency_resolution * exposure)

        timestamp = (
            datetime.datetime.strptime(
                paired_hdu["sig_caloff"].header["DATE-OBS"], "%Y-%m-%dT%H:%M:%S.%f"
            )
            .replace(tzinfo=datetime.timezone.utc)
            .timestamp()
        )
        tau = zenith_opacity.get_opacity(timestamp, frequency)
        elevation = paired_hdu["sig_caloff"].header["ELEVATIO"]
        estimated_noise_corrected = (
            estimated_noise
            * numpy.exp(tau / numpy.sin(numpy.deg2rad(elevation)))
            / eta_l
        )
        return estimated_noise_corrected


class PositionSwitchedCalibration(Calibration):

    @staticmethod
    def pair_up_rows(
        rows: pandas.DataFrame,
    ) -> list[dict[PairedScanName, pandas.Series]]:
        position_switched_rows = rows.query("PROCEDURE in ['OffOn', 'OnOff']")
        if len(rows) != len(position_switched_rows):
            loguru.logger.warning(
                f"Found {len(rows) - len(position_switched_rows)} rows using procedure that is not supported. Supported procedures are ['OffOn', 'OnOff']."
            )

        rows = position_switched_rows
        is_first_scan = (rows.PROCEDURE == "OffOn") == (rows.PROCSCAN == "OFF")
        rows["PAIRED_OFFSCAN"] = numpy.where(is_first_scan, rows.SCAN, rows.SCAN - 1)
        groups = rows.groupby(["SOURCE", "PAIRED_OFFSCAN", "SAMPLER"])
        paired_up_rows: list[dict[PairedScanName, pandas.Series]] = list()
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
                    dict(
                        ref_caloff=ref_caloff_,
                        ref_calon=ref_calon_,
                        sig_caloff=sig_caloff_,
                        sig_calon=sig_calon_,
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
