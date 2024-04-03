from mpi4py import MPI

from process_utils import (
    merge_and_print_results,
    process_lines,
    send_lines,
    single_process,
)

file = "../twitter-100gb.json"

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size > 1:
    if rank == 0:
        send_lines(file, comm)
        merge_and_print_results(comm)
    else:
        process_lines(comm)
else:
    single_process(file)
