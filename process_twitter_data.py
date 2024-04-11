import os
import sys
import time
from collections import defaultdict
from typing import Generator

from mpi4py import MPI

from process_utils import get_created_time, get_sentiment, to_day, to_hour


def read_file(filename: str, rank: int, size: int) -> Generator[str, None, None]:
    """read the file given current rank and size and generate lines"""
    file_size = os.path.getsize(filename)
    chunk_size = file_size // size
    start_pos = rank * chunk_size
    # end_pos = start_pos + chunk_size if rank != size - 1 else file_size
    bytes_read = 0
    with open(filename, "rb") as f:
        f.seek(start_pos)
        for line in f:
            if rank == 0 or bytes_read > 0:
                yield line.decode("utf-8")
                bytes_read += len(line)
            if bytes_read >= chunk_size:
                break
    # with open(filename, "rb") as f:
    #     if start_pos != 0:
    #         f.seek(start_pos - 1)
    #         prev_byte = f.read(1)
    #         if prev_byte.decode("utf-8") != "\n":
    #             f.readline()
    #     while f.tell() < end_pos:
    #         yield f.readline().decode("utf-8")


def process_lines(
    lines: Generator[str, None, None],
) -> tuple[dict[str, float], dict[str, float], dict[str, int], dict[str, int]]:
    """process the lines and return the results"""
    hour_sentiment: dict[str, float] = defaultdict(float)
    day_sentiment: dict[str, float] = defaultdict(float)
    hour_tweets: dict[str, int] = defaultdict(int)
    day_tweets: dict[str, int] = defaultdict(int)

    for line in lines:
        # the first and last line of the file do not end with ",\n", they are ignored
        if not line.endswith(",\n"):
            continue

        created_time = get_created_time(line)
        if created_time is None:
            continue
        hour = str(to_hour(created_time))
        day = str(to_day(created_time))
        sentiment = get_sentiment(line)

        hour_tweets[hour] += 1
        day_tweets[day] += 1
        if sentiment is not None:
            hour_sentiment[hour] += sentiment
            day_sentiment[day] += sentiment

    return hour_sentiment, day_sentiment, hour_tweets, day_tweets


def merge_results(
    results: list[
        tuple[dict[str, float], dict[str, float], dict[str, int], dict[str, int]]
    ]
) -> tuple[dict[str, float], dict[str, float], dict[str, int], dict[str, int]]:
    """merge the results"""
    merged_hour_sentiment: dict[str, float] = defaultdict(float)
    merged_day_sentiment: dict[str, float] = defaultdict(float)
    merged_hour_tweets: dict[str, int] = defaultdict(int)
    merged_day_tweets: dict[str, int] = defaultdict(int)
    for hour_sentiment, day_sentiment, hour_tweets, day_tweets in results:
        for hour, sentiment in hour_sentiment.items():
            merged_hour_sentiment[hour] += sentiment
        for day, sentiment in day_sentiment.items():
            merged_day_sentiment[day] += sentiment
        for hour, tweets in hour_tweets.items():
            merged_hour_tweets[hour] += tweets
        for day, tweets in day_tweets.items():
            merged_day_tweets[day] += tweets

    return (
        merged_hour_sentiment,
        merged_day_sentiment,
        merged_hour_tweets,
        merged_day_tweets,
    )


def print_results(
    hour_sentiment: dict[str, float],
    day_sentiment: dict[str, float],
    hour_tweets: dict[str, int],
    day_tweets: dict[str, int],
) -> None:
    """print the results"""
    print(f"happiest hour: {max(hour_sentiment, key=lambda k :hour_sentiment[k])}")
    print(f"happiest day: {max(day_sentiment, key=lambda k :day_sentiment[k])}")
    print(f"most active hour: {max(hour_tweets, key=lambda k :hour_tweets[k])}")
    print(f"most active day: {max(day_tweets, key=lambda k :day_tweets[k])}")


# read the filename from the command line
filename = sys.argv[1]

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

start_time = time.time()
lines = read_file(filename, rank, size)
result = process_lines(lines)
end_time = time.time()
print(f"rank {rank} take {end_time - start_time}s to process data")

results = comm.gather(result, root=0)
if rank == 0:
    start_time = time.time()
    assert results is not None
    merged_results = merge_results(results)
    end_time = time.time()
    print(f"rank {rank} take {end_time - start_time}s to merge results")
    print_results(*merged_results)
