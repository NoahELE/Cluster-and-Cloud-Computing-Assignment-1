import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

sns.set_theme(style="whitegrid")

time_1n1c = "39m32.343s"
time_1n8c = "9m55.108s"
time_2n8c = "10m47.546s"


def parse_time_as_seconds(time_str: str) -> float:
    minutes = float(time_str[: time_str.index("m")])
    seconds = float(time_str[time_str.index("m") + 1 : time_str.index("s")])
    return minutes * 60 + seconds


df = pl.DataFrame(
    {
        "config": ["1 node 1 core", "1 node 8 cores", "2 nodes 8 cores"],
        "time": [
            parse_time_as_seconds(time_1n1c),
            parse_time_as_seconds(time_1n8c),
            parse_time_as_seconds(time_2n8c),
        ],
    }
)

ax = sns.barplot(data=df, x="config", y="time")
ax.set_title("Execution time of jobs")
ax.set_xlabel("Configuration (nodes and cores)")
ax.set_ylabel("Time (s)")
plt.savefig("execution_time.png")
