# TODO: Find a better way to import src.metrics
import sys

import gpytorch
import numpy as np
import torch
from bayesian_benchmarks.data import Wilson_elevators
from gp_config import GPConfig
from gp_model import ExactGPModel
from sklearn.model_selection import train_test_split

sys.path.append("../")
import src.metrics as metrics  # noqa: E402
from utils.torch_utils import To_device  # noqa: E402

device = torch.device("cuda:0")
to_device = To_device(device)

data = Wilson_elevators()
X_train_full, X_val, Y_train_full, Y_val = train_test_split(
    data.X_train, data.Y_train, test_size=0.2, random_state=444
)
X_train_full, Y_train_full = data.X_train, data.Y_train  # Train on complete train-set
X_test = data.X_test
Y_test = data.Y_test

X_train_full = X_train_full.astype(np.float32)
Y_train_full = Y_train_full.astype(np.float32)
X_val = X_val.astype(np.float32)
Y_val = Y_val.astype(np.float32)

X_test, Y_test = X_test.astype(np.float32), Y_test.astype(np.float32)
X_train_full, Y_train_full, X_test, Y_test = to_device(
    (X_train_full, Y_train_full, X_test, Y_test)
)
X_val, Y_val = to_device((X_val, Y_val))
Y_train_full_gp, Y_test_gp = Y_train_full.reshape(-1), Y_test.reshape(-1)
Y_val_gp = Y_val.reshape(-1)


likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(X_train_full, Y_train_full_gp, likelihood)
model = model.to(device)

model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=GPConfig.gp_lr)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

losses = []
rmse_vals = []
msll_vals = []
for i in range(GPConfig.gp_n_epochs):
    optimizer.zero_grad()
    output = model(X_train_full)
    loss = -mll(output, Y_train_full_gp)
    loss.backward()
    losses.append(loss.item())
    optimizer.step()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        model.eval()
        pred = likelihood(model(X_val))

        rmse_val = metrics.root_mean_square_error(pred.mean, Y_val_gp)
        msll_val = metrics.mean_standardized_log_loss(
            pred.mean, pred.variance, Y_val_gp
        )
        rmse_vals += [rmse_val]
        msll_vals += [msll_val]
        print(
            "Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f   rmse: %.3f   msll: %.3f"
            % (
                i + 1,
                GPConfig.gp_n_epochs,
                loss.item(),
                model.covar_module.base_kernel.lengthscale.item(),
                model.likelihood.noise.item(),
                rmse_val,
                msll_val,
            )
        )
    model.train()


with torch.no_grad(), gpytorch.settings.fast_pred_var():
    model.eval()
    pred = likelihood(model(X_test))

    rmse_val = metrics.root_mean_square_error(pred.mean, Y_test_gp)
    msll_val = metrics.mean_standardized_log_loss(pred.mean, pred.variance, Y_test_gp)
    nlpd_val = metrics.negative_log_predictive_density(
        pred.mean.cpu(), pred.covariance_matrix.cpu(), Y_test_gp.cpu()
    )
    avg_covg = metrics.average_coverage_error(pred.mean, pred.stddev, Y_test_gp)

    print(
        "RMSE: ",
        rmse_val,
        "MSLL: ",
        msll_val,
        "NLPD: ",
        nlpd_val,
        "Avg Coverage: ",
        avg_covg,
    )
