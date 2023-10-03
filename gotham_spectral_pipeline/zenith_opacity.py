from .utils import datetime_parser

import datetime
import pathlib
import re
import typing

import loguru
import numpy
import numpy.typing
import scipy.interpolate

__all__ = ["ZenithOpacity"]


class ZenithOpacity:

    class Database:
        InterpolationFunction = typing.Callable[
            [numpy.typing.NDArray[numpy.floating]], numpy.typing.NDArray[numpy.floating]
        ]

        path: pathlib.Path
        loaded: bool

        _time: numpy.typing.NDArray[numpy.floating]
        _frequency: numpy.typing.NDArray[numpy.floating]
        _opacity: numpy.typing.NDArray[numpy.floating]
        _interpolation_functions: dict[tuple[float, float], list[InterpolationFunction]]

        def __init__(self, path: pathlib.Path):
            self.path = path
            self.loaded = False

        def _load(self):
            self._time = numpy.loadtxt(self.path / "time.txt")
            self._frequency = numpy.loadtxt(self.path / "frequency.txt") * 1e9
            self._opacity = numpy.loadtxt(self.path / "opacity.txt", unpack=True)
            self._interpolation_functions = dict()
            for line in open(self.path / "frequency_lists.txt", "r"):
                start = float(line.split(maxsplit=1)[0]) * 1e9
                stop = float(line.rsplit(maxsplit=1)[-1]) * 1e9
                lidx = numpy.searchsorted(self._frequency, start, side="left")
                ridx = numpy.searchsorted(self._frequency, stop, side="right")
                self._interpolation_functions[start, stop] = [
                    scipy.interpolate.CubicSpline(
                        self._frequency[lidx:ridx], opacity[lidx:ridx]
                    )
                    for opacity in self._opacity
                ]

            # Convert time from MJD to unix timestamp
            self._time = (
                self._time * 86400
                + datetime.datetime(
                    year=1858,
                    month=11,
                    day=17,
                    hour=0,
                    minute=0,
                    second=0,
                    tzinfo=datetime.timezone.utc,
                ).timestamp()
            )

            self.loaded = True

        @property
        def time(self) -> numpy.typing.NDArray[numpy.floating]:
            if not self.loaded:
                self._load()
            return self._time

        @property
        def frequency(self) -> numpy.typing.NDArray[numpy.floating]:
            if not self.loaded:
                self._load()
            return self._frequency

        @property
        def opacity(self) -> numpy.typing.NDArray[numpy.floating]:
            if not self.loaded:
                self._load()
            return self._opacity

        @property
        def interpolation_functions(
            self,
        ) -> dict[tuple[float, float], list[InterpolationFunction]]:
            if not self.loaded:
                self._load()
            return self._interpolation_functions

        def get_opacity(
            self, timestamp: float, frequency: numpy.typing.ArrayLike
        ) -> numpy.typing.NDArray[numpy.floating]:
            tidx = int(numpy.searchsorted(self.time, timestamp, side="left"))
            if self.time[tidx] == timestamp:
                return self._interpolate_opacity(tidx, frequency)
            left = self._interpolate_opacity(tidx - 1, frequency)
            right = self._interpolate_opacity(tidx, frequency)
            left_weight = self.time[tidx] - timestamp
            right_weight = timestamp - self.time[tidx - 1]
            total_weight = self.time[tidx] - self.time[tidx - 1]
            return (left * left_weight + right * right_weight) / total_weight

        def _interpolate_opacity(
            self, tidx: int, frequency: numpy.typing.ArrayLike
        ) -> numpy.typing.NDArray[numpy.floating]:
            frequency_array = numpy.array(frequency, dtype=float)
            opacity_array = numpy.full_like(frequency, numpy.nan)
            for (
                start,
                stop,
            ), interpolation_functions in self.interpolation_functions.items():
                lidx = numpy.searchsorted(frequency_array, start, side="left")
                ridx = numpy.searchsorted(frequency_array, stop, side="right")
                if lidx < ridx:
                    opacity_array[lidx:ridx] = interpolation_functions[tidx](
                        frequency_array[lidx:ridx]
                    )
            return opacity_array

    path: pathlib.Path
    databases: dict[tuple[int, int], Database]

    def __init__(self, path: str):
        self.path = pathlib.Path(path)
        self.databases = dict()

        timestamp_matcher = re.compile(r"Forecasts_(\d+)_(\d+)")
        for subdirectory in self.path.glob("Forecasts_*_*"):
            if subdirectory.is_file():
                continue

            result = timestamp_matcher.match(subdirectory.name)
            if result is None:
                loguru.logger.warning(
                    f"Cannot parse timestamps from {subdirectory = }. Ignoring..."
                )
                continue

            start, stop = map(int, result.groups(1))
            self.databases[start, stop] = ZenithOpacity.Database(subdirectory)

    def get_database(self, timestamp: float) -> Database | None:
        for (start, stop), database in self.databases.items():
            if start <= timestamp <= stop:
                return database
        return None

    def get_opacity(
        self, timestamp: float, frequency: numpy.typing.ArrayLike
    ) -> numpy.typing.NDArray[numpy.floating]:
        database = self.get_database(timestamp)
        if database is None:
            return numpy.full_like(frequency, numpy.nan)
        return database.get_opacity(timestamp, frequency)

    @staticmethod
    def generate_cleo_command(
        datetime_range: tuple[str, str],
        incremental_hour: int = 1,
        frequency_lists: list[numpy.typing.NDArray[numpy.floating]] | None = None,
        frequency_range: tuple[float, float] | None = None,
        no_caching: bool = True,
        force_remove: bool = False,
        overwrite: bool = False,
    ) -> str:
        """Generate one-liner command for generating opacity using CLEO on GBT machines

        Args:
            datetime_range (tuple[str, str]): Datetime range represented by a tuple of string expressed in ISO 8601 extended format, i.e. "yyyy-mm-ddThh:mm:ss[.mmmmmm]".
            incremental_hour (int, optional): Number of hours between each calculation. Defaults to 1.
            frequency_lists (list[numpy.typing.NDArray[numpy.floating]] | None, optional): List of arrays of frequency, each representing a list of frequency in a continous range. If None, default frequency lists that skips certain water and oxygen lines are used. Defaults to None.
            no_caching (bool, optional): Use no-caching option for more accurate but slower results. Defaults to True.

        Returns:
            str: Shell command for generating opacity using CLEO on GBT machines.
        """

        def snap_to_hour(dt: datetime.datetime) -> datetime.datetime:
            delta_t = datetime.timedelta(
                minutes=dt.minute, seconds=dt.second, microseconds=dt.microsecond
            )
            return dt - delta_t

        start_datetime, stop_datetime = map(datetime_parser, datetime_range)
        start_datetime = snap_to_hour(start_datetime)
        delta_t = datetime.timedelta(hours=incremental_hour)
        stop_datetime = (
            stop_datetime - (stop_datetime - start_datetime) % delta_t + delta_t
        )

        if frequency_lists is None:

            def generate_frequency_list(
                start: float,
                stop: float,
                no_epsilons_at_head: bool = False,
                no_epsilons_at_tail: bool = False,
            ) -> numpy.typing.NDArray[numpy.floating]:
                epsilons = numpy.geomspace(0.00001, 1.0, 21)

                start_linspace = int(numpy.floor(start)) + 1
                stop_linspace = int(numpy.ceil(stop)) - 1

                frequency_list: list[numpy.typing.ArrayLike] = []
                frequency_list.append(start)
                if not no_epsilons_at_head:
                    frequency_list.extend(start + epsilons)
                    start_linspace += 1
                if not no_epsilons_at_tail:
                    frequency_list.extend(stop - epsilons)
                    stop_linspace -= 1
                frequency_list.extend(
                    numpy.linspace(
                        start_linspace,
                        stop_linspace,
                        num=stop_linspace - start_linspace + 1,
                    )
                )
                frequency_list.append(stop)

                return numpy.sort(numpy.array(frequency_list))

            frequency_lists = [
                numpy.concatenate([
                    generate_frequency_list(1.0, 22.23508, no_epsilons_at_head=True),
                    generate_frequency_list(22.23508, 52.0),
                ]),
                generate_frequency_list(68.0, 116.0),
                generate_frequency_list(121.0, 171.0),
                generate_frequency_list(197.0, 235.0, no_epsilons_at_tail=True),
            ]

        if frequency_range is not None:
            start, stop = frequency_range
            filtered_frequency_lists: list[numpy.typing.NDArray[numpy.floating]] = []
            for frequency_list in frequency_lists:
                in_range = (start <= frequency_list) & (frequency_list <= stop)
                if any(in_range):
                    filtered_frequency_lists.append(frequency_list[in_range])
            frequency_lists = filtered_frequency_lists

        file_suffix = "_".join(
            map(
                str,
                [
                    int(start_datetime.timestamp()),
                    int(stop_datetime.timestamp()),
                ],
            )
        )

        directory_name = f"Forecasts_{file_suffix}"
        commands: list[str] = []

        remove_command = "rm"
        if force_remove:
            remove_command += " -f"
        if overwrite:
            commands.append(
                f"test ! -e {directory_name} || {remove_command} {directory_name}"
            )
        commands.append(f"test ! -e {directory_name}")

        commands.append(f"mkdir {directory_name}")
        commands.extend(
            [
                f"echo {' '.join(map(str, frequency_list))} >> {directory_name}/frequency_lists.txt"
                for frequency_list in frequency_lists
            ]
        )

        cleo_command_pieces = [
            "/home/cleoversions/Cleo/mainscreens/forecastsCmdLine.tcl",
            "-average",
            "-calculate OpacityTime",
            f"-freqList `cat {directory_name}/frequency_lists.txt`",
            f"-startTime '{start_datetime.strftime('%m/%d/%Y %H:%M:%S')}'",
            f"-stopTime '{stop_datetime.strftime('%m/%d/%Y %H:%M:%S')}'",
            f"-incrTime {incremental_hour}",
            f"-fileStamp {file_suffix}",
        ]
        if no_caching:
            cleo_command_pieces.append("-noCaching")
        commands.append(" ".join(cleo_command_pieces))

        commands.append(
            "("
            + " | ".join([
                f"head -n 1 {directory_name}/Transposed_time_avrg_{file_suffix}.txt",
                f"sed 's/timeListMJD //' > {directory_name}/time.txt",
            ])
            + ")"
        )

        commands.append(
            "("
            + " | ".join([
                f"tail -n +2 {directory_name}/Transposed_time_avrg_{file_suffix}.txt",
                f"sed 's/OpacityTime\\(.*\\)List_avrg.*/\\1/' > {directory_name}/frequency.txt",
            ])
            + ")"
        )
        commands.append(
            "("
            + " | ".join([
                f"tail -n +2 {directory_name}/Transposed_time_avrg_{file_suffix}.txt",
                f"sed 's/OpacityTime.*List_avrg \\(.*\\)/\\1/'",
                f"sed 's/ /\\t/g' > {directory_name}/opacity.txt",
            ])
            + ")"
        )
        commands.append(f"{remove_command} {directory_name}/*_{file_suffix}.txt")

        return " && ".join(commands)
