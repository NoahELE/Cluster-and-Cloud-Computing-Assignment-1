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
