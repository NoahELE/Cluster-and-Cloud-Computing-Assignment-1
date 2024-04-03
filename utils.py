from datetime import datetime
from typing import Any

import orjson


def get_lines(file: str) -> list[str]:
    """read the lines of data file (skip the first and last line)"""
    with open(file, encoding="utf-8") as f:
        # skip the first and last line
        lines = f.readlines()[1:-1]
        # remove trailing comma and newline
        lines = [line.rstrip(",\n") for line in lines]
        return lines


def parse_line(line: str) -> dict[str, Any]:
    """parse the line into a dict"""
    return orjson.loads(line)


def get_sentiment(row: dict[str, Any]) -> float | None:
    """get the sentiment from the row, return None if it does not exist"""
    try:
        sentiment = row["doc"]["data"]["sentiment"]
        if isinstance(sentiment, float) or isinstance(sentiment, int):
            return sentiment
        elif isinstance(sentiment, dict):
            return sentiment["score"]
        else:
            raise ValueError("Invalid sentiment type", sentiment)
    except KeyError:
        return None


def get_hour(row: dict[str, Any]) -> int | None:
    """get the hour from the row"""
    try:
        datetime_str = row["doc"]["data"]["created_at"]
        parsed = datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S.%fZ")
        return parsed.hour
    except KeyError:
        return None


def get_day(row: dict[str, Any]) -> int | None:
    """get the day from the row"""
    try:
        datetime_str = row["doc"]["data"]["created_at"]
        parsed = datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S.%fZ")
        return parsed.day
    except KeyError:
        return None
