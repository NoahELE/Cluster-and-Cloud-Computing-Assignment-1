import sys
import time
from collections import defaultdict

from mpi4py import MPI
from mpi4py.MPI import Intracomm

from process_utils import get_created_time, get_sentiment, to_day, to_hour


def single_process(filename: str) -> None:
    """process and print the results in a single process"""
    hour_sentiment: dict[str, float] = defaultdict(float)
    day_sentiment: dict[str, float] = defaultdict(float)
    hour_tweets: dict[str, int] = defaultdict(int)
    day_tweets: dict[str, int] = defaultdict(int)

    with open(filename, "rb") as f:
        # process the data line by line
        for line in f:
            line = line.decode("utf-8")
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


def send_lines(filename: str, comm: Intracomm) -> None:
    """send the lines of data file from rank 0 to other ranks"""
    size = comm.size
    ranks = range(1, size)
    i = 0
    with open(filename, "rb") as f:
        # send the lines to ranks 1 to size - 1
        for line in f:
            # send the line to the rank
            comm.send(line.decode("utf-8"), dest=ranks[i])
            # move to next rank
            i = (i + 1) % len(ranks)
    # send None to all ranks to indicate the end of the data
    for r in ranks:
        comm.send(None, dest=r)


def process_lines(comm: Intracomm) -> None:
    """process the lines of data sent by rank 0"""
    hour_sentiment: dict[str, float] = defaultdict(float)
    day_sentiment: dict[str, float] = defaultdict(float)
    hour_tweets: dict[str, int] = defaultdict(int)
    day_tweets: dict[str, int] = defaultdict(int)

    # process the lines until None is received
    while (line := comm.recv(source=0)) is not None:
        created_time = get_created_time(line)
        if created_time is None:
            continue
        hour = str(to_day(created_time))
        day = str(to_hour(created_time))
        sentiment = get_sentiment(line)

        hour_tweets[hour] += 1
        day_tweets[day] += 1
        if sentiment is not None:
            hour_sentiment[hour] += sentiment
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
    if rank == 0:
        # send lines to other ranks
        start_time = time.time()
        send_lines(filename, comm)
        end_time = time.time()
        print(f"Time taken for sending on rank {rank}: {end_time - start_time}s")

        # merge the results from other ranks
        start_time = time.time()
        merge_and_print_results(comm)
        end_time = time.time()
        print(f"Time taken for merging on rank {rank}: {end_time - start_time}s")
    else:
        # process the lines sent by rank 0
        start_time = time.time()
        process_lines(comm)
        end_time = time.time()
        print(f"Time taken for processing on rank {rank}: {end_time - start_time}s")
