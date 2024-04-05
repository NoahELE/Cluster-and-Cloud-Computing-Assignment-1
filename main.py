import numpy as np
import time
from mpi4py import MPI

from utils import read_file, get_date, get_sentiment, happiest_hour_calculation, happiest_day_calculation, most_active_hour_calculation, most_active_day_calculation, log_time


FILE = 'twitter-1mb.json'

# Initialize MPI
comm = MPI.COMM_WORLD

# Get the rank of the current process
rank = comm.Get_rank()

# Get the total number of processes
size = comm.Get_size()


# log the start time
start_time = time.time()


    
hour_sentiments = np.zeros((12, 31, 24), dtype=float)
hour_tweets = np.zeros((12, 31, 24), dtype=int)

def process_tweets():
     # Read the file and process each tweet
    for tweet_line in read_file(FILE, rank, size):
        # Process each tweet_line here
        datetime = get_date(tweet_line)
        sentiment_score = get_sentiment(tweet_line)

        if (datetime is not None):
           # print(datetime)
            hour_tweets[datetime[0] -1, datetime[1] -1, datetime[2]] += 1
            
            if (sentiment_score is not None):
                hour_sentiments[datetime[0] -1, datetime[1] -1, datetime[2]] += sentiment_score
        



process_tweets()


# print the running time
log_time('Time taken for reading and processing: ', start_time)

# gather the results from all the processes to the root process

gathered_hour_sentiments = comm.gather(hour_sentiments, root=0)
gathered_hour_tweets = comm.gather(hour_tweets, root=0)

if comm.rank == 0:  # Only in the root process
    hour_sentiments_final = np.sum(gathered_hour_sentiments, axis=0)
    hour_tweets_final = np.sum(gathered_hour_tweets, axis=0)
    happiest_hour_calculation(hour_sentiments_final)
    happiest_day_calculation(hour_sentiments_final)
    most_active_hour_calculation(hour_tweets_final)
    most_active_day_calculation(hour_tweets_final)

# print the running time
log_time('Total time taken: ', start_time)
