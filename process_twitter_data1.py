import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Generator

import orjson
from mpi4py import MPI

from process_utils import get_day, get_hour, get_sentiment


def count_lines(filename: str) -> int:
    with open(filename, "r", encoding="utf8") as f:
        return sum(1 for _ in f)


def read_lines(
    filename: str, total_lines: int, rank: int, size: int
) -> Generator[str, None, None]:
    chunk_size = total_lines // size
    start = rank * chunk_size
    end = start + chunk_size if rank != size - 1 else total_lines
    with open(filename, "r", encoding="utf8") as f:
        for _ in range(start):
            next(f)
        for _ in range(start, end):
            yield next(f)


comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

filename = sys.argv[1]

if rank == 0:
    start = time.time()
    total_lines = count_lines(filename)
    print(f"Time for counting lines on rank {rank}: {time.time() - start}s")
else:
    total_lines = 0
total_lines = comm.bcast(total_lines, root=0)


start = time.time()
lines = read_lines(filename, total_lines, rank, size)
hour_sentiment: dict[datetime, float] = defaultdict(float)
day_sentiment: dict[datetime, float] = defaultdict(float)
hour_tweets: dict[datetime, int] = defaultdict(int)
day_tweets: dict[datetime, int] = defaultdict(int)
for line in lines:
    if not line.endswith(",\n"):
        continue
    line = line.rstrip(",\n")
    row: dict[str, Any] = orjson.loads(line)
    day = get_day(row)
    hour = get_hour(row)
    sentiment = get_sentiment(row)

    if day is not None:
        day_tweets[day] += 1
        if sentiment is not None:
            day_sentiment[day] += sentiment
    if hour is not None:
        hour_tweets[hour] += 1
        if sentiment is not None:
            hour_sentiment[hour] += sentiment
print(f"Time for processing lines on rank {rank}: {time.time() - start}s")

hour_sentiment_list = comm.gather(hour_sentiment, root=0)
day_sentiment_list = comm.gather(day_sentiment, root=0)
hour_tweets_list = comm.gather(hour_tweets, root=0)
day_tweets_list = comm.gather(day_tweets, root=0)
if rank == 0:
    start = time.time()
    assert (
        hour_sentiment_list is not None
        and day_sentiment_list is not None
        and hour_tweets_list is not None
        and day_tweets_list is not None
    )
    merged_hour_sentiment: dict[datetime, float] = defaultdict(float)
    merged_day_sentiment: dict[datetime, float] = defaultdict(float)
    merged_hour_tweets: dict[datetime, int] = defaultdict(int)
    merged_day_tweets: dict[datetime, int] = defaultdict(int)
    for hour_sentiment, day_sentiment, hour_tweets, day_tweets in zip(
        hour_sentiment_list, day_sentiment_list, hour_tweets_list, day_tweets_list
    ):
        for hour, sentiment in hour_sentiment.items():
            merged_hour_sentiment[hour] += sentiment
        for day, sentiment in day_sentiment.items():
            merged_day_sentiment[day] += sentiment
        for hour, tweets in hour_tweets.items():
            merged_hour_tweets[hour] += tweets
        for day, tweets in day_tweets.items():
            merged_day_tweets[day] += tweets
    print(
        f"happiest hour: {max(merged_hour_sentiment, key=lambda k :merged_hour_sentiment[k])}"
    )
    print(
        f"happiest day: {max(merged_day_sentiment, key=lambda k :merged_day_sentiment[k])}"
    )
    print(
        f"most active hour: {max(merged_hour_tweets, key=lambda k :merged_hour_tweets[k])}"
    )
    print(
        f"most active day: {max(merged_day_tweets, key=lambda k :merged_day_tweets[k])}"
    )
    print(f"Time for merging results on rank {rank}: {time.time() - start}s")
