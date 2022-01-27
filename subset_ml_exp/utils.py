import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.cluster import kmeans_plusplus
from scipy.stats import qmc

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


def split_stations_d2(stations_df, n_train, n_test, random_state):
    stations_coordinates = stations_df[["latitude", "longtitude"]].values
    _, row_indices = kmeans_plusplus(
        stations_coordinates, n_clusters=n_test, random_state=random_state
    )
    stations_test = list(stations_df.iloc[row_indices].index)
    stations_train_full = [
        s_id for s_id in stations_df.index if s_id not in stations_test
    ]
    stations_train, statitions_pool = train_test_split(
        stations_train_full, train_size=n_train, random_state=random_state
    )
    return map(sorted, (stations_train, stations_test, statitions_pool))


def split_stations_lhs(stations_df, n_train, n_test, random_state):
    stations_coordinates = stations_df[["latitude", "longtitude"]].values
    l_bounds = np.min(stations_coordinates, axis = 0)
    u_bounds = np.max(stations_coordinates, axis = 0)

    lhs_sampler = qmc.LatinHypercube(d = 2, centered=False, seed = random_state)
    lhs_samples = lhs_sampler.random(n = n_test)
    lhs_samples = qmc.scale(lhs_samples, l_bounds, u_bounds)

    stations_test = []
    for add_iter in range(n_test):
        nearest_station_id, nearest_station_dist = 0, np.inf
        for station_id in stations_df.index.unique():
            if station_id in stations_test:
                continue
            station_coordinates = stations_df.loc[station_id][["latitude", "longtitude"]].values
            dist = np.sum(np.square(lhs_samples[add_iter] - station_coordinates))
            if dist < nearest_station_dist:
                nearest_station_id, nearest_station_dist = station_id, dist
        stations_test.append(nearest_station_id)
    
    stations_train_full = [s_id for s_id in stations_df.index if s_id not in stations_test]
    stations_train, statitions_pool = train_test_split(
        stations_train_full, train_size=n_train, random_state=random_state
    )
    return map(sorted, (stations_train, stations_test, statitions_pool))


def split_stations(
    stations_df, n_train=6, n_test=6, strategy="random", random_state=42
):
    assert len(stations_df) >= (n_train + n_test)
    if strategy == "random":
        return split_stations_random(
            list(stations_df.index), n_train, n_test, random_state
        )
    if strategy == "d2":
        return split_stations_d2(stations_df, n_train, n_test, random_state)
    if strategy == "lhs":
        return split_stations_lhs(stations_df, n_train, n_test, random_state)
    else:
        raise Exception(f"Only available strategies are 'random', 'd2', 'lhs'")

def plot_stations(
    train_stations,
    test_stations,
    pool_stations,
    stations_df,
    fig_title,
    strategy,
    newly_added_station_id=None,
):
    ax = plt.subplot(111)
    if newly_added_station_id:
        ax.scatter(
            stations_df.loc[
                [
                    newly_added_station_id,
                ]
            ]["longtitude"],
            stations_df.loc[
                [
                    newly_added_station_id,
                ]
            ]["latitude"],
            marker="P",
            color="y",
            label="new train",
        )
        train_stations = train_stations.copy()
        train_stations.remove(newly_added_station_id)
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
    # Reference:
    # https://stackoverflow.com/questions/741877/how-do-i-tell-matplotlib-that-i-am-done-with-a-plot
    plt.close()


def To_device(device):
    def to_device(input, device=device):
        if isinstance(input, (tuple, list)):
            return [to_device(x) for x in input]
        if type(input) == np.ndarray:
            input = torch.tensor(input)
        return input.to(device, non_blocking=True)

    return to_device
