import typing

import loguru
import numpy
import pandas

PairedScanName = typing.Literal["ref_caloff", "ref_calon", "sig_caloff", "sig_calon"]


class Calibration:
    pass


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
