import re
from datetime import datetime
from typing import Any


def get_sentiment(row: dict[str, Any]) -> float | None:
    """get the sentiment from the row, return None if it does not exist"""
    try:
        sentiment = row["doc"]["data"]["sentiment"]
        if isinstance(sentiment, float) or isinstance(sentiment, int):
            return sentiment
        # elif isinstance(sentiment, dict):
        #     return sentiment["score"]
        else:
            # raise ValueError("Invalid sentiment type", sentiment)
            return None
    except KeyError:
        return None


def get_hour(row: dict[str, Any]) -> datetime | None:
    """get the hour from the row"""
    try:
        datetime_str = row["doc"]["data"]["created_at"]
        parsed = datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S.%fZ")
        return parsed.replace(minute=0, second=0, microsecond=0)
    except KeyError:
        return None


def get_day(row: dict[str, Any]) -> datetime | None:
    """get the day from the row"""
    try:
        datetime_str = row["doc"]["data"]["created_at"]
        parsed = datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S.%fZ")
        return parsed.replace(hour=0, minute=0, second=0, microsecond=0)
    except KeyError:
        return None


# NOTE: The following code is not used in the final version of the code


sentiment_pattern = re.compile(r'"sentiment":\s*(-?\d+\.?\d*)')
created_time_pattern = re.compile(
    r'"created_at":\s*"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z)"'
)


def get_sentiment_re(line: str) -> float | None:
    """get the sentiment from the row, return None if it does not exist"""
    m = sentiment_pattern.search(line)
    if m is None:
        return None
    else:
        return float(m.group(1))


def get_created_time_re(line: str) -> datetime | None:
    m = created_time_pattern.search(line)
    if m is None:
        return None
    else:
        return datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S.%fZ")


def to_day(date: datetime) -> datetime:
    return date.replace(hour=0, minute=0, second=0, microsecond=0)


def to_hour(date: datetime) -> datetime:
    return date.replace(minute=0, second=0, microsecond=0)
