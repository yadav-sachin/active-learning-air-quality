import pandas as pd
import numpy as np
import gpytorch
import torch
from gp_model import ExactGPModel, GPConfig
from utils import plot_stations, split_stations
from sklearn.preprocessing import StandardScaler

random_seed = 7593
np.random.seed(random_seed)
# Split Beijing Stations Randomly
beijing_stations_csv_path = "./data/Beijing/Station.txt"
stations_df_req_cols = ["station_id", "latitude", "longtitude"]
beijing_stations_df = pd.read_csv(
    beijing_stations_csv_path, usecols=stations_df_req_cols
)
beijing_stations_df = beijing_stations_df.set_index("station_id")
beijing_stations_df = beijing_stations_df.sort_values(
    by=[
        "station_id",
    ],
    ascending=[
        True,
    ],
)
beijing_stations = list(beijing_stations_df.index)
(
    beijing_stations_train,
    beijing_stations_test,
    beijing_stations_pool,
) = split_stations(
    beijing_stations, n_train=6, n_test=6, strategy="random", random_state=random_seed
)

plot_stations(
    beijing_stations_train,
    beijing_stations_test,
    beijing_stations_pool,
    beijing_stations_df,
    "Beijing Dataset",
    strategy="random",
)

# 24 observations by each station in each day
window_size = 24

data_df_req_cols = ["station_id", "time", "PM25_AQI_value"]
data_df = pd.read_csv("data/Beijing/CrawledData.txt", usecols=data_df_req_cols)
data_df = data_df.join(beijing_stations_df, on="station_id", how="left")
data_df["time"] = pd.to_datetime(data_df["time"])
data_df = data_df.set_index("time")


start_timestamp = pd.to_datetime("2013-11-09 00:00:01")
end_timestamp = pd.to_datetime("2013-11-20 00:00:01")  # Not inclusive
desired_timerange_mask = (data_df.index > start_timestamp) & (
    data_df.index < end_timestamp
)
data_df = data_df[desired_timerange_mask]
data_df = data_df.sort_values(by=["time", "station_id"], ascending=[True, True])
all_unique_timestamps = data_df.index.unique()

# Loop for each day
for day_start_timestamp_index in range(0, len(all_unique_timestamps), window_size):

    lat_long_scaler = StandardScaler()
    pm25_scaler = StandardScaler()

    current_required_timestamps = all_unique_timestamps[
        day_start_timestamp_index : day_start_timestamp_index + window_size
    ]
    current_data_df = data_df.loc[current_required_timestamps]

    current_data_train_df = current_data_df[
        current_data_df["station_id"].isin(beijing_stations_train)
    ]
    current_data_test_df = current_data_df[
        current_data_df["station_id"].isin(beijing_stations_test)
    ]
    current_data_pool_df = current_data_df[
        current_data_df["station_id"].isin(beijing_stations_pool)
    ]

    lat_long_scaler.fit(current_data_train_df[["latitude", "longtitude"]].values)
    pm25_scaler.fit(current_data_train_df[["PM25_AQI_value"]].values)

    for current_timestamp in current_data_df.index.unique():
        train_x = current_data_train_df.loc[
            current_timestamp, ["latitude", "longtitude"]
        ].values
        pool_x = current_data_pool_df.loc[
            current_timestamp, ["latitude", "longtitude"]
        ].values
        test_x = current_data_test_df.loc[
            current_timestamp, ["latitude", "longtitude"]
        ].values

        train_x, test_x, pool_x = map(
            lat_long_scaler.transform, (train_x, test_x, pool_x)
        )

        train_y = current_data_train_df.loc[
            current_timestamp, ["PM25_AQI_value"]
        ].values
        train_y = pm25_scaler.transform(train_y)

        test_y = current_data_test_df.loc[current_timestamp, "PM25_AQI_value"].values

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        gp_model = ExactGPModel(
            train_x, train_y, likelihood, ard_num_dims=train_x.shape[1]
        )

        gp_model.train()
        likelihood.train()

        gp_optimizer = torch.optim.Adam(gp_model.parameters(), lr=GPConfig.lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

        for i in range(GPConfig.n_train_iter):
            gp_optimizer.zero_grad()
            output = gp_model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            print(
                "Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f"
                % (
                    i + 1,
                    GPConfig.n_train_iter,
                    loss.item(),
                    gp_model.covar_module.base_kernel.lengthscale.item(),
                    gp_model.likelihood.noise.item(),
                )
            )
            gp_optimizer()

        pred_y = likelihood(gp_model(test_x))
        pred_y = pm25_scaler.transform(pred_y)

        pred_var_pool = gp_model(pool_x).variance

        break

    break
