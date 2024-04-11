import os
import sys
from collections import defaultdict
from typing import Generator

from mpi4py import MPI

from process_utils import get_created_time, get_sentiment, to_day, to_hour


def single_process(filename: str) -> None:
    """process and print the results in a single process"""
    hour_sentiment: dict[str, float] = defaultdict(float)
    day_sentiment: dict[str, float] = defaultdict(float)
    hour_tweets: dict[str, int] = defaultdict(int)
    day_tweets: dict[str, int] = defaultdict(int)

    with open(filename, encoding="utf-8") as f:
        # process the data line by line
        for line in f:
            # the first and last line of the file do not end with ",\n"
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

    # print the results
    print(f"happiest hour: {max(hour_sentiment, key=lambda k :hour_sentiment[k])}")
    print(f"happiest day: {max(day_sentiment, key=lambda k :day_sentiment[k])}")
    print(f"most active hour: {max(hour_tweets, key=lambda k :hour_tweets[k])}")
    print(f"most active day: {max(day_tweets, key=lambda k :day_tweets[k])}")


def read_file(filename: str, rank: int, size: int) -> Generator[str, None, None]:
    """read the file and return the lines as a list"""
    file_size = os.path.getsize(filename)
    chunk_size = file_size // size
    start_pos = rank * chunk_size
    end_pos = start_pos + chunk_size if rank != size - 1 else file_size
    with open(filename, "r", encoding="utf-8") as f:
        if start_pos != 0:
            f.seek(start_pos)
            f.readline()
        while f.tell() < end_pos:
            line = f.readline()
            if line.endswith(",\n"):
                yield line
            else:
                break


# read the filename from the command line
filename = sys.argv[1]

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

if size == 1:
    # if only one process, run the single process code
    single_process(filename)
else:
    raise NotImplementedError("Only one process is supported")
