import wandb
import torch
import numpy as np
import sklearn.gaussian_process as skgp
import sklearn.utils.validation as skval
import scipy.stats as stat

import utils
import constants

kernels = {
    "rbf": skgp.kernels.RBF,
    "matern": skgp.kernels.Matern,
    "rat_quad": skgp.kernels.RationalQuadratic,
    "periodic": skgp.kernels.ExpSineSquared,
}

# Standard Gaussian Process regression model, this class is
# a wrapper for the sci-kit learn implementation
#
# Note that a GP model needs access to the train data at test-time, so the model needs
# to be trained and tested in one run (set both train and test to 1 in config).
class GP:
    def __init__(self, config):
        assert config["gp_kernel"] in kernels, "Unknown kernel: '{}'".format(
                config["gp_kernel"])

        # Add on WhiteKernel to optimize noise variance parameter
        kernel = kernels[config["gp_kernel"]]() + skgp.kernels.WhiteKernel()

        self.device = config["device"] # For working with pytorch

        # alpha = 0 since we use a WhiteKernel (noise variance is learned)
        # See for example: https://scikit-learn.org/stable/modules/gaussian_process.html#gpr-with-noise-level-estimation
        self.gp = skgp.GaussianProcessRegressor(kernel=kernel, alpha=0.,
                n_restarts_optimizer=config["opt_restarts"])

    def train(self, train_set, config, val_func=None):
        # Train
        self.gp.fit(train_set.x, train_set.y)

        # Validate
        val_func(self, epoch_i=1)

    def get_pdf(self, x, **kwargs):
        skval.check_is_fitted(self.gp, msg="GP is not fitted, impossible to get pdf")
        if type(x) == torch.Tensor:
            x = x.numpy()

        predict_x = np.atleast_2d(x)
        mean, std = self.gp.predict(predict_x, return_std=True)
        return utils.get_gaussian_pdf(mean[0,0], std[0])

    def sample(self, xs, **kwargs):
        skval.check_is_fitted(self.gp, msg="GP is not fitted, impossible to sample")

        # xs is always torch tensor
        # Fix since x is unnecessarily repeated
        # (this is ineffective, but doesn't impact the actual model)
        unique_x, counts = torch.unique_consecutive(xs, return_counts=True, dim=0)
        n_samples = counts[0].item() # Assume all counts the same

        unique_x = unique_x.numpy() # to numpy
        # random state = None means use numpy random,
        # which is already seeded at test time
        samples = self.gp.sample_y(unique_x, n_samples=n_samples, random_state=None)
        samples_torch = torch.tensor(samples, device=self.device, dtype=torch.float)

        # Put y-dim last and flatten samples for each x
        reshaped_samples = torch.transpose(samples_torch, 1, 2).flatten(0,1)

        return reshaped_samples

    def eval(self, dataset, config, **kwargs):
        skval.check_is_fitted(self.gp, msg="GP is not fitted, impossible to get pdf")

        # Targets to numpy
        y_np = dataset.y.numpy()

        # Compute log-likelihood
        means, stds = self.gp.predict(dataset.x.numpy(), return_std=True)
        covs = np.power(stds, 2)

        logpdfs = [stat.multivariate_normal.logpdf(y, mean=m, cov=c)
                for y, m,c in zip(y_np, means, covs)] # Slow, but ok for this
        ll = np.mean(logpdfs)

        # Compute mean absolute error
        abs_diff = np.abs(means - y_np) # Means are also medians because Gaussian
        mae = np.mean(np.sum(abs_diff, axis=1))

        return {"ll": ll, "mae": mae}


def build_gp(config):
    return GP(config)

