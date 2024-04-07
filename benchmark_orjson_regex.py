import re
import time
from typing import Any

import orjson

pattern = re.compile(r'"sentiment":\s*(-?\d+\.?\d*)')


def get_sentiment1(tweet_line):
    if "sentiment" not in tweet_line:
        return None
    # pattern = r'"sentiment":\s*(-?\d+\.?\d*)'
    match = re.search(pattern, tweet_line)

    if match:
        sentiment = float(match.group(1))
    else:
        return None

    return sentiment


def get_sentiment2(row: dict[str, Any]) -> float | None:
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


with open("../twitter-50mb.json", "r", encoding="utf8") as f:
    lines = f.readlines()[1:-1]

lines = [line.strip(",\n") for line in lines]

start = time.time()
for line in lines:
    row = orjson.loads(line)
    get_sentiment2(row)
    # get_sentiment2(row)
    # get_sentiment2(row)
    # get_sentiment2(row)
    # get_sentiment2(row)
print("Time taken for get_sentiment from process_utils.py: ", time.time() - start)

start = time.time()
for line in lines:
    get_sentiment1(line)
    # get_sentiment1(line)
    # get_sentiment1(line)
    # get_sentiment1(line)
    # get_sentiment1(line)
print("Time taken for get_sentiment from utils.py: ", time.time() - start)
