import orjson

from utils import get_hour, get_lines, get_sentiment

lines = get_lines("twitter-1mb.json")


def test_get_sentiment() -> None:
    row = orjson.loads(lines[0])
    sentiment = get_sentiment(row)
    assert sentiment == 0.7142857142857143


def test_get_hour() -> None:
    row = orjson.loads(lines[0])
    hour = get_hour(row)
    assert hour == 3
