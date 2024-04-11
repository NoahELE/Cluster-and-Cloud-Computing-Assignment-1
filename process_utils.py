import re
from datetime import datetime

sentiment_pattern = re.compile(r'"sentiment":\s*(-?\d+\.?\d*)')
created_time_pattern = re.compile(
    r'"created_at":\s*"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z)"'
)


def get_sentiment(line: str) -> float | None:
    """get the sentiment from the row, return None if it does not exist"""
    m = sentiment_pattern.search(line)
    if m is None:
        return None
    else:
        return float(m.group(1))


def get_created_time(line: str) -> datetime | None:
    m = created_time_pattern.search(line)
    if m is None:
        return None
    else:
        return datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S.%fZ")


def to_day(date: datetime) -> datetime:
    return date.replace(hour=0, minute=0, second=0, microsecond=0)


def to_hour(date: datetime) -> datetime:
    return date.replace(minute=0, second=0, microsecond=0)
