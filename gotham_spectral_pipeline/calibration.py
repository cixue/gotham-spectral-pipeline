import typing

import astropy.io.fits
import loguru
import numpy
import numpy.typing
import pandas

PairedScanName = typing.Literal["ref_caloff", "ref_calon", "sig_caloff", "sig_calon"]


class Calibration:

    @staticmethod
    def _verify_paired_hdu(
        paired_hdu: dict[PairedScanName, astropy.io.fits.PrimaryHDU]
    ) -> bool:
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

    @staticmethod
    def get_system_temperature(
        paired_hdu: dict[PairedScanName, astropy.io.fits.PrimaryHDU]
    ) -> float | None:
        if not Calibration._verify_paired_hdu(paired_hdu):
            return None

        Tcals: set[float] = set()
        for _, hdu in paired_hdu.items():
            Tcals.add(hdu.header["TCAL"])

        Tcal = Tcals.pop()
        if len(Tcals) != 0:
            loguru.logger.warning(
                f"Tcal's are not identical in the paired up HDU. Using {Tcal = }. The rest are {Tcals}."
            )

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

    @staticmethod
    def get_antenna_temperature(
        paired_hdu: dict[PairedScanName, astropy.io.fits.PrimaryHDU]
    ) -> numpy.typing.NDArray[numpy.floating] | None:
        if not Calibration._verify_paired_hdu(paired_hdu):
            return None

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
