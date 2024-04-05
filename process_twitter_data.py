from mpi4py import MPI

from process_utils import (
    merge_and_print_results,
    process_lines,
    send_lines,
    single_process,
)

filename = "../twitter-100gb.json"

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

if size == 1:
    # if only one process, run the single process code
    single_process(filename)
else:
    # else run the parallel code
    if rank == 0:
        send_lines(filename, comm)
        merge_and_print_results(comm)
    else:
        process_lines(comm)
