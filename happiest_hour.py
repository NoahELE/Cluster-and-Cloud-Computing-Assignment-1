import statistics

from mpi4py import MPI

from utils import get_hour, get_lines, get_sentiment, parse_line

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

lines = get_lines("../twitter-1mb.json")

chunk_size = len(lines) // size
start = rank * chunk_size
end = start + chunk_size if rank != size - 1 else len(lines)

rows = [parse_line(line) for line in lines[start:end]]

hour_sentiment: dict[int, list[float]] = {}
for row in rows:
    sentiment = get_sentiment(row)
    if sentiment is None:
        continue
    hour = get_hour(row)
    if hour is None:
        continue
    if hour not in hour_sentiment:
        hour_sentiment[hour] = []
    hour_sentiment[hour].append(sentiment)

gathered: list[dict[int, list[float]]] | None = comm.gather(hour_sentiment, root=0)

# gather data on rank 0
if rank == 0:
    merged: dict[int, list[float]] = {}
    assert gathered is not None
    print("gathered: ", gathered[0])
    for g in gathered:
        for k, v in g.items():
            if k not in merged:
                merged[k] = []
            merged[k].extend(v)
    print("result: ", max(merged.keys(), key=lambda k: statistics.mean(merged[k])))
