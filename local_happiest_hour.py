from utils import get_hour, get_lines, get_sentiment, parse_line

lines = get_lines("../twitter-50mb.json")
rows = [parse_line(line) for line in lines]

hour_sentiment: dict[int, float] = {}
for row in rows:
    sentiment = get_sentiment(row)
    if sentiment is None:
        continue
    hour = get_hour(row)
    if hour is None:
        continue
    if hour not in hour_sentiment:
        hour_sentiment[hour] = 0
    hour_sentiment[hour] += sentiment


print(
    "happiest hour: ",
    max(hour_sentiment.keys(), key=lambda k: hour_sentiment[k]),
)
