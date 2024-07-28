from __future__ import annotations

import datetime
import re
import os
import pathlib
import typing
import warnings

import loguru

builtin_showwarning = None


def capture_builtin_warnings():
    global builtin_showwarning
    builtin_showwarning = warnings.showwarning

    def showwarning(message, *_, **__):
        loguru.logger.warning(message)

    warnings.showwarning = showwarning


class LogLimiter:
    _rate_setting: dict[str, float]
    _last_emit: dict[tuple[str, str], datetime.datetime]
    _stats: dict[tuple[str, str], dict[typing.Literal["silenced", "emitted"], int]]
    _regex_number_matcher: re.Pattern
    _do_not_filter: bool

    def __init__(
        self,
        prefix: str,
        logs_directory: pathlib.Path = pathlib.Path("."),
        **rate_setting: float,
    ):
        all_levels = [
            "TRACE",
            "DEBUG",
            "INFO",
            "SUCCESS",
            "WARNING",
            "ERROR",
            "CRITICAL",
        ]
        self._rate_setting = {level: 0 for level in all_levels}
        self._rate_setting.update(rate_setting)
        self._last_emit = {}
        self._stats = {}
        self._regex_number_matcher = re.compile(r"[+-]?\d+")
        self._do_not_filter = False

        loguru.logger.remove()
        os.makedirs(logs_directory, exist_ok=True)
        for i, level in enumerate(all_levels):
            filename = logs_directory / f"{prefix}.{i}.{level.lower()}.log"
            loguru.logger.add(
                filename,
                level=0,
                filter=self._filter_generator(
                    [self._filter_by_level(level), self._filter_by_rate]
                ),
                mode="w",
            )

    def _filter_generator(self, filters):
        def combined_filter(record: loguru.Record) -> bool:
            return all(map(lambda f: f(record), filters))

        return combined_filter

    def _filter_by_level(self, level: str) -> typing.Callable[[loguru.Record], bool]:
        def filter_by_level(record: loguru.Record) -> bool:
            return record["level"].name == level

        return filter_by_level

    def _filter_by_rate(self, record: loguru.Record) -> bool:
        if self._do_not_filter:
            return True
        level = record["level"].name
        message = self._regex_number_matcher.sub("<number>", record["message"])
        key = (level, message)
        time = record["time"]
        if key not in self._stats:
            self._stats[key] = {}
            self._stats[key]["emitted"] = 0
            self._stats[key]["silenced"] = 0

        if (
            key not in self._last_emit
            or (time - self._last_emit[key]).total_seconds() > self._rate_setting[level]
        ):
            self._last_emit[key] = time
            self._stats[key]["emitted"] += 1
            return True
        else:
            self._stats[key]["silenced"] += 1
            return False

    def log_silence_report(self):
        self._do_not_filter = True
        for (level, message), stats in self._stats.items():
            loguru.logger.info(
                f"Stats for ({level = }, {message = !r}).\n"
                f"Silenced {stats['silenced']} times. Emitted {stats['emitted']} times.\n"
                f"Total {stats['silenced'] + stats['emitted']} times."
            )
        self._do_not_filter = False
