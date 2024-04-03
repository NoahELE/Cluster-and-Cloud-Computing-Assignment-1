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
        return row["doc"]["data"]["sentiment"]
    except KeyError:
        return None


def get_hour(row: dict[str, Any]) -> int:
    """get the hour from the row"""
    datetime_str = row["doc"]["data"]["created_at"]
    parsed = datetime.fromisoformat(datetime_str)
    return parsed.hour
