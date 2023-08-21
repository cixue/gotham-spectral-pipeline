import datetime

import numpy
import numpy.typing


class ZenithOpacity:

    @staticmethod
    def generate_cleo_command(
        datetime_range: tuple[str, str],
        incremental_hour: int = 1,
        frequency_lists: list[numpy.typing.NDArray[numpy.floating]] | None = None,
        no_caching: bool = True,
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

        start_datetime, stop_datetime = (
            datetime.datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S.%f").replace(
                tzinfo=datetime.timezone.utc
            )
            for dt in datetime_range
        )
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
        commands.append(f"rm {directory_name}/*_{file_suffix}.txt")

        return " && ".join(commands)
