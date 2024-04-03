from mpi4py import MPI

from utils import get_day, get_lines, get_sentiment, parse_line

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

lines = get_lines("../twitter-100gb.json")

chunk_size = len(lines) // size
start = rank * chunk_size
end = start + chunk_size if rank != size - 1 else len(lines)

rows = [parse_line(line) for line in lines[start:end]]

day_sentiment: dict[int, float] = {}
for row in rows:
    sentiment = get_sentiment(row)
    if sentiment is None:
        continue
    day = get_day(row)
    if day is None:
        continue
    if day not in day_sentiment:
        day_sentiment[day] = 0
    day_sentiment[day] += sentiment

# gather data on rank 0
gathered: list[dict[int, float]] | None = comm.gather(day_sentiment, root=0)
if rank == 0:
    merged: dict[int, float] = {}
    assert gathered is not None
    for g in gathered:
        for k, v in g.items():
            if k not in merged:
                merged[k] = 0
            merged[k] += v
    print("happiest day: ", max(merged.keys(), key=lambda k: (merged[k])))
