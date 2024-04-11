import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

time_1n1c_list = ["19m40.081s", "20m58.300s"]
time_1n8c_list = ["3m7.192s", "3m49.165s"]
time_2n8c_list = ["2m51.745s", "3m57.090s"]


def parse_time_as_seconds(time_str: str) -> float:
    minutes = float(time_str[: time_str.index("m")])
    seconds = float(time_str[time_str.index("m") + 1 : time_str.index("s")])
    return minutes * 60 + seconds


df = pl.DataFrame(
    {
        "config": ["1 node 1 core", "1 node 8 cores", "2 nodes 8 cores"],
        "algorithm1": [
            parse_time_as_seconds(time_1n1c_list[0]),
            parse_time_as_seconds(time_1n8c_list[0]),
            parse_time_as_seconds(time_2n8c_list[0]),
        ],
        "algorithm2": [
            parse_time_as_seconds(time_1n1c_list[1]),
            parse_time_as_seconds(time_1n8c_list[1]),
            parse_time_as_seconds(time_2n8c_list[1]),
        ],
    }
)

df = df.melt(id_vars=["config"], value_name="time", variable_name="algorithm")
print(df)

sns.set_theme(style="darkgrid")
ax = sns.barplot(data=df, x="config", y="time", hue="algorithm")
ax.set_title("Execution Time of Jobs")
ax.set_xlabel("Configuration (nodes and cores)")
ax.set_ylabel("Time (s)")
plt.savefig("execution_time.png")
# plt.show()
