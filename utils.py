from datetime import datetime
import numpy as np
import re


def read_file(file_path, rank, size):

    file_size = os.path.getsize(file_path)
    chunk_size = file_size // size
    start = rank * chunk_size
    bytes_read = 0

    with open(file_path, 'r', encoding='utf-8') as file:
        file.seek(start, 0)

        for line in file:
            if rank == 0 or bytes_read > 0:
                yield line
            bytes_read += len(line)

            if bytes_read >= chunk_size:
                break

def get_date(tweet_line):
    
    if "created_at" not in tweet_line:
        return None
    datetime_string = tweet_line.split("created_at")[1]
    pattern = r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.\d{3}Z'
    match = re.search(pattern, datetime_string)
    if match:
        datetime_string = match.group()
    if datetime_string is None:
        return None
    dt = datetime.strptime(datetime_string, "%Y-%m-%dT%H:%M:%S.%fZ")
    datetime_tuple = (dt.month, dt.day, dt.hour)
    return datetime_tuple
    


def get_sentiment(tweet_line):
    if "sentiment" not in tweet_line:
        return None
    pattern = r'"sentiment":\s*(-?\d+\.\d+)'
    match = re.search(pattern, tweet_line)

    if match:
        sentiment = float(match.group(1))
    else:
        return None

    return sentiment
    


def happiest_hour_calculation(hour_sentiments):
    # Find the index of the maximum sentiment score
    happiest_hour = np.unravel_index(np.argmax(hour_sentiments), hour_sentiments.shape)
    max_sentiment = np.max(hour_sentiments)

    # Extract the month, day, and hour from the happiest_hour_index
    month = happiest_hour[0] + 1
    day = happiest_hour[1] + 1
    hour = happiest_hour[2]

     # Print the maximum sentiment score for information
    print(f"Maximum sentiment score: {max_sentiment}")
    print(f"Happiest hour: {hour}-{hour+1} on {month}-{day}")

def happiest_day_calculation(hour_sentiments):
    happiest_day = np.unravel_index(np.argmax(np.sum(hour_sentiments, axis=2)), np.sum(hour_sentiments, axis=2).shape)
    max_sentiment = np.max(np.sum(hour_sentiments, axis=2))
    
    month = happiest_day[0] + 1
    day = happiest_day[1] + 1
    print(f"Maximum sentiment score: {max_sentiment}")
    print(f"Happiest day: {month}-{day}")

def most_active_hour_calculation(hour_tweets):
    most_active_hour = np.unravel_index(np.argmax(hour_tweets), hour_tweets.shape)
    number_of_tweets = np.max(hour_tweets)

    month = most_active_hour[0] + 1
    day = most_active_hour[1] + 1
    hour = most_active_hour[2]
    print(f"Number of tweets: {number_of_tweets}")
    print(f"Most active hour: {hour}-{hour+1} on {month}-{day}")
    

def most_active_day_calculation(hour_tweets):
    most_active_day = np.unravel_index(np.argmax(np.sum(hour_tweets, axis=2)), np.sum(hour_tweets, axis=2).shape)
    number_of_tweets = np.max(np.sum(hour_tweets, axis=2))

    month = most_active_day[0] + 1
    day = most_active_day[1] + 1
    print(f"Number of tweets: {number_of_tweets}")
    print(f"Most active day: {month}-{day}")

def log_time(message, start_time):
    end_time = time.time()
    running_time = end_time - start_time
    print(message, running_time)