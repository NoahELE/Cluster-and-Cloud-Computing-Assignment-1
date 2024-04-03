from mpi4py import MPI

from utils import merge_and_print_results, process_lines, send_lines

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    send_lines("../twitter-100gb.json", comm)
    merge_and_print_results(comm)
else:
    process_lines(comm)
