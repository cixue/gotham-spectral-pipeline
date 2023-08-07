import collections
import io
import pathlib
import re

import loguru
import pandas


class SDFits:
    path: pathlib.Path
    path_to_index: pathlib.Path
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
        if self.path.is_dir():
            self.path_to_index = self.path / (self.path.name + ".index")
        else:
            self.path_to_index, self.path = self.path, self.path.parent
        if not self.path_to_index.exists():
            raise FileNotFoundError(f"{self.path_to_index} not found.")

        blocks = SDFits._read_blocks(self.path_to_index)
        self.header = SDFits._parse_header(blocks.pop("header"))
        self.rows = SDFits._parse_rows(blocks.pop("rows"))
        if blocks:
            loguru.logger.warning(f"Found unused blocks: {list(blocks.keys())}.")

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
