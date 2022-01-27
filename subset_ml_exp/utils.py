import matplotlib.pyplot as plt

plt.style.use("seaborn")
from sklearn.model_selection import train_test_split


def split_stations_random(stations, n_train, n_test, random_state):
    stations_train_full, stations_test = train_test_split(
        stations, test_size=n_test, random_state=random_state
    )
    stations_train, statitions_pool = train_test_split(
        stations_train_full, train_size=n_train, random_state=random_state
    )
    return map(sorted, (stations_train, stations_test, statitions_pool))


def split_stations(stations, n_train=6, n_test=6, strategy="random", random_state=42):
    if strategy == "random":
        return split_stations_random(stations, n_train, n_test, random_state)


def plot_stations(
    train_stations,
    test_stations,
    pool_stations,
    stations_df,
    fig_title,
    strategy="random",
):
    ax = plt.subplot(111)
    ax.scatter(
        stations_df.loc[train_stations]["longtitude"],
        stations_df.loc[train_stations]["latitude"],
        marker="d",
        color="k",
        label="train",
    )
    ax.scatter(
        stations_df.loc[test_stations]["longtitude"],
        stations_df.loc[test_stations]["latitude"],
        marker="o",
        color="r",
        label=f"test, {strategy} sampling",
    )
    ax.scatter(
        stations_df.loc[pool_stations]["longtitude"],
        stations_df.loc[pool_stations]["latitude"],
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
