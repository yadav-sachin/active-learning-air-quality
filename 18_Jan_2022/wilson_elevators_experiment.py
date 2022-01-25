# TODO: Find a better way to import src.metrics
import sys

import gpytorch

# import numpy as np
import torch
from gp_config import GPConfig
from gp_model import ExactGPModel
from bayesian_benchmarks.data import Wilson_elevators

sys.path.append("../")
import src.data_utils as data_utils  # noqa: E402
import src.metrics as metrics  # noqa: E402
from utils.torch_utils import To_device  # noqa: E402

device = torch.device("cuda:0")
to_device = To_device(device)

X_train_full, X_val, X_test, Y_train_full_gp, Y_val_gp, Y_test_gp = data_utils.process_data(
    Wilson_elevators(), to_device, train_val_split=False
)


likelihood = gpytorch.likelihoods.GaussianLikelihood()
covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())
model = ExactGPModel(X_train_full, Y_train_full_gp, likelihood, covar_module)
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
