import os
from typing import Generator


def read_file(filename: str, rank: int, size: int) -> Generator[str, None, None]:
    """read the file given current rank and size and generate lines"""
    file_size = os.path.getsize(filename)
    chunk_size = file_size // size
    start_pos = rank * chunk_size
    end_pos = start_pos + chunk_size if rank != size - 1 else file_size
    with open(filename, "rb") as f:
        if start_pos != 0:
            f.seek(start_pos - 1)
            prev_byte = f.read(1)
            if prev_byte.decode("utf-8") != "\n":
                f.readline()
        while f.tell() < end_pos:
            yield f.readline().decode("utf-8")


print(list(read_file("../twitter-50mb.json", 0, 1))[:10])
