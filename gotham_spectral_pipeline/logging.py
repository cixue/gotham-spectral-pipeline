from __future__ import annotations

import datetime
import re
import os
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
    _silenced: dict[tuple[str, str], int]
    _regex_number_matcher: re.Pattern
    _do_not_filter: bool

    def __init__(self, **rate_setting: float):
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
        self._silenced = {}
        self._regex_number_matcher = re.compile(r"[+-]?\d+")
        self._do_not_filter = False

        loguru.logger.remove()
        os.makedirs("gsp_logs", exist_ok=True)
        for i, level in enumerate(all_levels):
            loguru.logger.add(
                f"gsp_logs/{i}.{level.lower()}.log",
                level=0,
                filter=self._filter_generator(
                    [self._filter_by_level(level), self._filter_by_rate]
                ),
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
        if (
            key not in self._last_emit
            or (time - self._last_emit[key]).total_seconds() > self._rate_setting[level]
        ):
            self._last_emit[key] = time
            self._silenced[key] = 0
            return True
        else:
            self._silenced[key] += 1
            return False

    def log_silence_report(self):
        self._do_not_filter = True
        for (level, message), silenced in self._silenced.items():
            loguru.logger.info(
                f"Silenced ({level = }, {message = !r}) for {silenced} times."
            )
        self._do_not_filter = False
