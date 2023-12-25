import ast
import collections
import io
import pathlib
import re

import astropy.io.fits
import loguru
import numpy
import pandas

__all__ = ["SDFits"]


class SDFits:
    path: pathlib.Path
    header: dict[str, str]
    rows: pandas.DataFrame

    def __init__(self, path: str):
        """SDFits index file parser

        Args:
            path (str): Path to the SDFits directory or index file, e.g. ".../AGBT18B_007_16.raw.vegas/" or ".../AGBT18B_007_16.raw.vegas/AGBT18B_007_16.raw.vegas.index".

        Raises:
            FileNotFoundError: Raised if the index file cannot be located.
        """
        self.path = pathlib.Path(path)

        if self.path.is_dir() or self.path.name.endswith(".index"):
            if self.path.is_dir():
                path_to_index = self.path / (self.path.name + ".index")
            else:
                path_to_index, self.path = self.path, self.path.parent
            if not path_to_index.exists():
                raise FileNotFoundError(f"{path_to_index} not found.")

            blocks = SDFits._read_blocks(path_to_index)
            self.header = SDFits._parse_header(blocks.pop("header"))
            self.rows = SDFits._parse_rows(blocks.pop("rows"))
            if blocks:
                loguru.logger.warning(f"Found unused blocks: {list(blocks.keys())}.")
            return

        if self.path.name.endswith(".fits"):
            path_to_fits, self.path = self.path, self.path.parent
            self.rows = self._build_rows_from_fits(path_to_fits)
            return

        loguru.logger.error("Unsupported file extension: {self.path}")

    @staticmethod
    def _read_blocks(path_to_index: pathlib.Path) -> dict[str, list[str]]:
        block_tag_matcher = re.compile("^\[(.+)\]$")
        blocks = collections.defaultdict(list)
        block_tag: (str | None) = None
        with open(path_to_index, "r") as f:
            for line in f:
                result = block_tag_matcher.match(line)
                if result:
                    block_tag = result.group(1)
                    continue
                if block_tag is None:
                    loguru.logger.warning("Found line before the first block.")
                    continue
                blocks[block_tag].append(line)
        return blocks

    @staticmethod
    def _parse_header(lines: list[str]) -> dict[str, str]:
        header = dict()
        for line in lines:
            key, value = (kv.strip() for kv in line.split("=", maxsplit=1))
            if not key:
                loguru.logger.warning("Found empty key.")
                continue
            if key in header:
                loguru.logger.warning(f"Found repeated key in header: {key}.")
                continue
            header[key] = value
        return header

    @staticmethod
    def _parse_rows(lines: list[str]) -> pandas.DataFrame:
        lines[0] = lines[0].replace("#INDEX#", " INDEX ")
        column_header_matcher = re.compile("\W*\w+")
        widths = [
            len(result.group()) for result in column_header_matcher.finditer(lines[0])
        ]
        return pandas.read_fwf(io.StringIO("".join(lines)), widths=widths)

    @staticmethod
    def _build_rows_from_fits(path_to_fits: pathlib.Path) -> pandas.DataFrame:
        with astropy.io.fits.open(path_to_fits, "readonly") as hdulist:
            single_dish_extension = None
            for extension, hdu in enumerate(hdulist):
                if not isinstance(hdu, astropy.io.fits.hdu.table.BinTableHDU):
                    continue
                if hdu.name != "SINGLE DISH":
                    continue
                if single_dish_extension is not None:
                    loguru.logger.error(
                        "More than one BinTableHDU named 'SINGLE DISH' is found."
                    )
                    return
                single_dish_extension = extension

            if single_dish_extension is None:
                loguru.logger.error("BinTableHDU named 'SINGLE DISH' is not found.")
                return

            hdu = hdulist[extension]
            data_column_index = hdu.columns.names.index("DATA") + 1
            ignored_column = [
                "DATA",
                f"TDIM{data_column_index}",
                f"TUNIT{data_column_index}",
            ]

            nrows = hdu.data.size
            rows = pandas.DataFrame()
            rows["INDEX"] = numpy.arange(nrows)
            rows["FILE"] = numpy.full(nrows, path_to_fits.name)
            rows["EXT"] = single_dish_extension
            rows["ROW"] = numpy.arange(nrows)
            for column in hdu.columns:
                if column.name in ignored_column:
                    continue
                data = hdu.data[column.name]
                if data.dtype.byteorder != '=':
                    data = data.byteswap(inplace=True).newbyteorder()
                rows[column.name] = data
            return rows

    def get_hdu_from_row(self, row: pandas.Series) -> astropy.io.fits.PrimaryHDU:
        """Get scan data represented by a record in the SDFits file.

        Args:
            row (pandas.Series): A record in the index file.

        Returns:
            astropy.io.fits.PrimaryHDU: HDU object containing the scan data.
        """
        path = self.path / row["FILE"]
        extension = row["EXT"]
        row_number = row["ROW"]
        with astropy.io.fits.open(path, "readonly") as hdulist:
            hdu = hdulist[extension]
            data_column_index = hdu.columns.names.index("DATA") + 1
            ignored_column = [
                "DATA",
                f"TDIM{data_column_index}",
                f"TUNIT{data_column_index}",
            ]

            scan = hdu.data[row_number]
            data_dimension = ast.literal_eval(scan[f"TDIM{data_column_index}"])
            data = scan["DATA"].reshape(data_dimension[::-1])

            header = astropy.io.fits.Header()
            for column in hdu.columns:
                if column.name in ignored_column:
                    continue
                try:
                    if len(column.name) > 8:
                        header[f"HIERARCH {column.name}"] = scan[column.name]
                    else:
                        header[column.name] = scan[column.name]
                except Exception as error:
                    loguru.logger.warning(
                        f"An {error = } is raised while processing {column = }. Ignoring..."
                    )
            return astropy.io.fits.PrimaryHDU(data=data, header=header)

    def get_hdulist_from_rows(
        self, rows: pandas.DataFrame
    ) -> list[astropy.io.fits.PrimaryHDU]:
        """Get a list of scan data represented by records in the SDFits file.

        Args:
            row (pandas.DataFrame): Multiple records in the index file.

        Returns:
            list[astropy.io.fits.PrimaryHDU]: A list of HDU objects containing the scan data.
        """
        return [self.get_hdu_from_row(row) for _, row in rows.iterrows()]

    def get_hdulist(self) -> list[astropy.io.fits.PrimaryHDU]:
        """Get all scan data in the SDFits file.

        Returns:
            list[astropy.io.fits.PrimaryHDU]: A list of HDU objects containing all scan data.
        """
        return self.get_hdulist_from_rows(self.rows)
