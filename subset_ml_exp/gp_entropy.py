import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils import plot_stations

# from sklearn.preprocessing import StandardScaler

np.random.seed(75943)

beijing_stations_csv_path = "./data/Beijing/Station.txt"
beijing_stations_df = pd.read_csv(beijing_stations_csv_path)

beijing_stations_train_full_df, beijing_stations_test_df = train_test_split(
    beijing_stations_df, test_size=6
)
beijing_stations_train_df, beijing_stations_pool_df = train_test_split(
    beijing_stations_train_full_df, train_size=6
)

plot_stations(
    beijing_stations_train_df,
    beijing_stations_test_df,
    beijing_stations_pool_df,
    "Beijing Dataset",
)
