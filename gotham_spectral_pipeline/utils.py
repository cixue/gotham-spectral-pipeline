import datetime
import functools
import typing

__all__ = [
    "datetime_parser",
    "lru_cache",
]

_P = typing.ParamSpec("_P")
_R = typing.TypeVar("_R")


@typing.overload
def lru_cache(func: typing.Callable[_P, _R], /, **kwargs) -> typing.Callable[_P, _R]:
    ...


@typing.overload
def lru_cache(
    **kwargs,
) -> typing.Callable[[typing.Callable[_P, _R]], typing.Callable[_P, _R]]:
    ...


def lru_cache(func: typing.Callable[_P, _R] | None = None, /, **kwargs):
    if func is not None:
        return functools.lru_cache(func, **kwargs)
    return functools.lru_cache(**kwargs)


def datetime_parser(dt: str) -> datetime.datetime:
    return datetime.datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S.%f").replace(
        tzinfo=datetime.timezone.utc
    )
