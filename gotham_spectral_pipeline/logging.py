from __future__ import annotations

import datetime
import re
import sys
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
        self._rate_setting = {
            level: 0
            for level in [
                "TRACE",
                "DEBUG",
                "INFO",
                "SUCCESS",
                "WARNING",
                "ERROR",
                "CRITICAL",
            ]
        }
        self._rate_setting.update(rate_setting)
        self._last_emit = {}
        self._silenced = {}
        self._regex_number_matcher = re.compile(r"[+-]?\d+")
        self._do_not_filter = False

        loguru.logger.remove()
        loguru.logger.add(sys.stdout, level=0, filter=self._filter)

    def _filter(self, record: loguru.Record) -> bool:
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
