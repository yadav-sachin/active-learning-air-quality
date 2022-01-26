import matplotlib.pyplot as plt

plt.style.use("seaborn")


def plot_stations(train_stations_df, test_stations_df, pool_stations_df, fig_title):
    ax = plt.subplot(111)
    ax.scatter(
        train_stations_df["longtitude"],
        train_stations_df["latitude"],
        marker="d",
        color="k",
        label="train",
    )
    ax.scatter(
        test_stations_df["longtitude"],
        test_stations_df["latitude"],
        marker="o",
        color="r",
        label="test, random sampling",
    )
    ax.scatter(
        pool_stations_df["longtitude"],
        pool_stations_df["latitude"],
        marker="*",
        color="g",
        label="pool",
    )
    # Reference: https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.grid(color="w")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(fig_title)
    plt.savefig(f"./assets/figs/{fig_title}.png")
