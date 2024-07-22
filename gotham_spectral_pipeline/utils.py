import datetime
import functools
import typing

import numpy
import numpy.typing

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


def searchsorted_nearest(
    a: numpy.typing.NDArray[numpy.floating], v: numpy.typing.NDArray[numpy.floating]
):
    right = numpy.searchsorted(a, v)
    left = right - 1
    right[right >= a.size] = a.size - 1
    left[left < 0] = 0
    nearest = numpy.where(v - a[left] < a[right] - v, left, right)
    (out_of_bound,) = numpy.where((v < a[0]) | (v > a[-1]))
    return nearest, out_of_bound
