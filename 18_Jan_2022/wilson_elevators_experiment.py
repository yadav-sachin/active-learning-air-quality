# TODO: Find a better way to import src.metrics
import sys

import gpytorch
import numpy as np
import torch
from bayesian_benchmarks.data import Wilson_elevators
from gp_config import GPConfig
from gp_model import ExactGPModel

sys.path.append("../")
import src.metrics as metrics  # noqa: E402
from utils.torch_utils import To_device  # noqa: E402

device = torch.device("cuda:1")
to_device = To_device(device)

data = Wilson_elevators()
X_train_full = data.X_train
Y_train_full = data.Y_train
X_test = data.X_test
Y_test = data.Y_test

X_train_full = X_train_full.astype(np.float32)
Y_train_full = Y_train_full.astype(np.float32)

X_test, Y_test = X_test.astype(np.float32), Y_test.astype(np.float32)
X_train_full, Y_train_full, X_test, Y_test = to_device(
    (X_train_full, Y_train_full, X_test, Y_test)
)
Y_train_full_gp, Y_test_gp = Y_train_full.reshape(-1), Y_test.reshape(-1)


likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(X_train_full, Y_train_full_gp, likelihood)
model = model.to(device)

model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=GPConfig.gp_lr)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

losses = []
for i in range(GPConfig.gp_n_epochs):
    optimizer.zero_grad()
    output = model(X_train_full)
    loss = -mll(output, Y_train_full_gp)
    loss.backward()
    print(
        "Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f"
        % (
            i + 1,
            GPConfig.gp_n_epochs,
            loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item(),
        )
    )
    losses.append(loss.item())
    optimizer.step()


with torch.no_grad(), gpytorch.settings.fast_pred_var():
    model.eval()
    pred = likelihood(model(X_test))

    print(pred.mean.shape, Y_test_gp.shape)

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
