import bayesian_benchmarks.data as bb_data
import numpy as np
from sklearn.model_selection import train_test_split


def process_data(
    dataset, to_device_func, train_val_split=False, val_size=0.2, random_state=444
):
    data = dataset

    X_train_full, X_val, Y_train_full, Y_val = train_test_split(
        data.X_train, data.Y_train, test_size=val_size, random_state=random_state
    )
    if not train_val_split:
        X_train_full, Y_train_full = (
            data.X_train,
            data.Y_train,
        )  # Train on complete train-set
    X_test = data.X_test
    Y_test = data.Y_test

    X_train_full = X_train_full.astype(np.float32)
    Y_train_full = Y_train_full.astype(np.float32)
    X_val = X_val.astype(np.float32)
    Y_val = Y_val.astype(np.float32)

    X_test, Y_test = X_test.astype(np.float32), Y_test.astype(np.float32)
    X_train_full, Y_train_full, X_test, Y_test = to_device_func(
        (X_train_full, Y_train_full, X_test, Y_test)
    )
    X_val, Y_val = to_device_func((X_val, Y_val))
    Y_train_full_gp, Y_test_gp = Y_train_full.reshape(-1), Y_test.reshape(-1)
    Y_val_gp = Y_val.reshape(-1)

    return X_train_full, X_val, X_test, Y_train_full_gp, Y_val_gp, Y_test_gp
