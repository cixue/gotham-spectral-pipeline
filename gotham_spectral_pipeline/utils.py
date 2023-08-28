import datetime


def datetime_parser(dt: str) -> datetime.datetime:
    return datetime.datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S.%f").replace(
        tzinfo=datetime.timezone.utc
    )
