import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

time_1n1c_list = ["39m32.343s", "27m10.711s", "24m15.404s"]
time_1n8c_list = ["9m55.108s", "9m39.067s", "8m52.014s"]
time_2n8c_list = ["10m47.546s", "9m13.522s", "8m17.379s"]


def parse_time_as_seconds(time_str: str) -> float:
    minutes = float(time_str[: time_str.index("m")])
    seconds = float(time_str[time_str.index("m") + 1 : time_str.index("s")])
    return minutes * 60 + seconds


df = pl.DataFrame(
    {
        "config": ["1 node 1 core", "1 node 8 cores", "2 nodes 8 cores"],
        "algorithm1_time": [
            parse_time_as_seconds(time_1n1c_list[0]),
            parse_time_as_seconds(time_1n8c_list[0]),
            parse_time_as_seconds(time_2n8c_list[0]),
        ],
        "algorithm2_time": [
            parse_time_as_seconds(time_1n1c_list[1]),
            parse_time_as_seconds(time_1n8c_list[1]),
            parse_time_as_seconds(time_2n8c_list[1]),
        ],
        "algorithm3_time": [
            parse_time_as_seconds(time_1n1c_list[2]),
            parse_time_as_seconds(time_1n8c_list[2]),
            parse_time_as_seconds(time_2n8c_list[2]),
        ],
    }
)

df = df.melt(id_vars=["config"], value_name="time", variable_name="algorithm")
print(df)

# sns.set_theme(style="whitegrid")
ax = sns.barplot(data=df, x="config", y="time", hue="algorithm")
ax.set_title("Execution time of jobs")
ax.set_xlabel("Configuration (nodes and cores)")
ax.set_ylabel("Time (s)")
plt.savefig("execution_time.png")
# plt.show()
