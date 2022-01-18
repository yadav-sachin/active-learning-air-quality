import torch
from scipy.stats import multivariate_normal, norm


def root_mean_square_error(y_hat, y):
    """
    Root Mean Square Error.
    y and y_hat would be reshaped into a 1D tensor before computing RMSE
    """
    with torch.no_grad():
        y_hat = y_hat.reshape(-1)
        y = y.reshape(-1)
        return torch.sqrt(torch.mean(torch.square(y_hat - y))).item()


def mean_standardized_log_loss(y_hat, y_hat_var, y):
    """
    MSLL.
    Helpful as RMSE does not take into account
    the standard deviation in case of predictive model
    Reference: https://issueexplorer.com/issue/scikit-learn/scikit-learn/21665

    Note: Make sure to include noise variance in the y_hat_var.
    """
    with torch.no_grad():
        y, y_hat, y_hat_var = y.reshape(-1), y_hat.reshape(-1)
        y_hat_var = y_hat_var.reshape(-1)

        term1 = torch.log(2 * torch.pi * y_hat_var)
        term2 = torch.square(y - y_hat) / y_hat_var
        return ((term1 + term2) / 2).mean().item()


def negative_log_predictive_density(y_hat, cov, y):
    """
    Negative Log Predictive Density
    Negative of log of probability density of normal[y_hat, cov] at y

    Note: Make sure to add noise variance in the diagonal of the cov matrix.
    """
    with torch.no_grad():
        y_hat, y = y_hat.reshape(-1), y.reshape(-1)
        rv = multivariate_normal(mean=y_hat, cov=cov)
        return -rv.logpdf(y_hat).item()


def average_coverage_error(y_hat, y_hat_std, y):
    """
    As per the normal distribution,
    the fraction of points within z-score * std away from mean;
    should be a certain fraction value corresponding to that z-score.
    Coverage error is the absolute difference
    in the fraction and expected fraction.

    For z-score = 2, the fraction is 0.9545

    For Avearage Coverage error, we take mean of coverage error
    for confidence level/fraction 0.05, 0.10, 0.15, 0.20, ... 0.95
    """
    with torch.no_grad():
        y_hat, y_hat_std = y_hat.reshape(-1), y_hat_std.reshape(-1)
        y = y.reshape(-1)

        confidence_levels = torch.arange(0.05, 1.0, 0.05)
        coverage_error = 0

        # TODO: Vectorise the below operation
        for confidence_level in confidence_levels:
            # Reference1:
            # https://stackoverflow.com/questions/20864847/probability-to-z-score-and-vice-versa
            # Reference2:
            # https://stackoverflow.com/questions/60699836/how-to-use-norm-ppf
            z_score = norm.ppf((1 + confidence_level) / 2)
            upper_lim = y_hat + z_score * y_hat_std
            lower_lim = y_hat - z_score * y_hat_std
            fraction = ((y_hat <= upper_lim) * (y_hat >= lower_lim)).mean()
            coverage_error += torch.abs(confidence_level - fraction).item()

        return coverage_error / len(confidence_levels)
