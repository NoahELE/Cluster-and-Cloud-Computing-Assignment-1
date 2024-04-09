import sys
import time
from collections import defaultdict
from datetime import datetime

import orjson
from mpi4py import MPI
from mpi4py.MPI import Intracomm

from process_utils import get_day, get_hour, get_sentiment


def single_process(filename: str) -> None:
    """process and print the results in a single process"""
    hour_sentiment: dict[datetime, float] = defaultdict(float)
    day_sentiment: dict[datetime, float] = defaultdict(float)
    hour_tweets: dict[datetime, int] = defaultdict(int)
    day_tweets: dict[datetime, int] = defaultdict(int)

    with open(filename, encoding="utf-8") as f:
        # process the data line by line
        for line in f:
            # the first and last line of the file do not end with ",\n"
            if not line.endswith(",\n"):
                continue

            line = line.rstrip(",\n")
            row = orjson.loads(line)
            sentiment = get_sentiment(row)
            hour = get_hour(row)
            day = get_day(row)

            if hour is not None:
                hour_tweets[hour] += 1
                if sentiment is not None:
                    hour_sentiment[hour] += sentiment
            if day is not None:
                day_tweets[day] += 1
                if sentiment is not None:
                    day_sentiment[day] += sentiment

    # print the results
    print(f"happiest hour: {max(hour_sentiment, key=lambda k :hour_sentiment[k])}")
    print(f"happiest day: {max(day_sentiment, key=lambda k :day_sentiment[k])}")
    print(f"most active hour: {max(hour_tweets, key=lambda k :hour_tweets[k])}")
    print(f"most active day: {max(day_tweets, key=lambda k :day_tweets[k])}")


def send_lines(filename: str, comm: Intracomm) -> None:
    """send the lines of data file from rank 0 to other ranks"""
    size = comm.size
    ranks = range(1, size)
    i = 0
    with open(filename, encoding="utf-8") as f:
        # send the lines to ranks 1 to size - 1
        for line in f:
            comm.send(line, dest=ranks[i])
            i = (i + 1) % len(ranks)
    # send None to all ranks to indicate the end of the data
    for r in ranks:
        comm.send(None, dest=r)


def process_lines(comm: Intracomm) -> None:
    """process the lines of data sent by rank 0"""
    hour_sentiment: dict[datetime, float] = defaultdict(float)
    day_sentiment: dict[datetime, float] = defaultdict(float)
    hour_tweets: dict[datetime, int] = defaultdict(int)
    day_tweets: dict[datetime, int] = defaultdict(int)

    # process the lines until None is received
    while (line := comm.recv(source=0)) is not None:
        # the first and last line of the file do not end with ",\n"
        if not line.endswith(",\n"):
            continue

        line = line.rstrip(",\n")
        row = orjson.loads(line)
        sentiment = get_sentiment(row)
        hour = get_hour(row)
        day = get_day(row)

        if hour is not None:
            hour_tweets[hour] += 1
            if sentiment is not None:
                hour_sentiment[hour] += sentiment
        if day is not None:
            day_tweets[day] += 1
            if sentiment is not None:
                day_sentiment[day] += sentiment

    # send the results to rank 0
    comm.send((hour_sentiment, day_sentiment, hour_tweets, day_tweets), dest=0)


def merge_and_print_results(comm: Intracomm) -> None:
    """merge the results from all ranks"""
    size = comm.size
    # receive the results from all ranks
    results = []
    for _ in range(1, size):
        results.append(comm.recv())

    # merge the results
    merged_hour_sentiment: dict[datetime, float] = defaultdict(float)
    merged_day_sentiment: dict[datetime, float] = defaultdict(float)
    merged_hour_tweets: dict[datetime, int] = defaultdict(int)
    merged_day_tweets: dict[datetime, int] = defaultdict(int)
    for hour_sentiment, day_sentiment, hour_tweets, day_tweets in results:
        for hour, sentiment in hour_sentiment.items():
            merged_hour_sentiment[hour] += sentiment
        for day, sentiment in day_sentiment.items():
            merged_day_sentiment[day] += sentiment
        for hour, tweets in hour_tweets.items():
            merged_hour_tweets[hour] += tweets
        for day, tweets in day_tweets.items():
            merged_day_tweets[day] += tweets

    # print the results
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


# read the filename from the command line
filename = sys.argv[1]

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

if size == 1:
    # if only one process, run the single process code
    single_process(filename)
else:
    # else run the parallel code
    start = time.time()
    if rank == 0:
        send_lines(filename, comm)
    else:
        process_lines(comm)
    print(f"Time taken for process on rank {rank}: {time.time() - start}s")

    start = time.time()
    if rank == 0:
        merge_and_print_results(comm)
        print(f"Time taken for merging on rank {rank}: {time.time() - start}s")
