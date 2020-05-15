import torch.nn as nn
import torch
import math

from models.nn import NN
import utils

# Neural network heteroskedastic regression model, models p(y|x) as an isotropic gaussian
# Predicts both the mean and variance of a gaussian
# Equal to p(y|x) = N(Y|NN1(x), NN2(x)), (NN1 and NN2 share hidden layers)
class NNHeteroskedastic(NN):
    def __init__(self, config):
        super().__init__(config)

        self.y_dim = config["y_dim"]

    def compute_dims(self, config, spec):
        # Simplest case, takes just x in
        spec["in_dim"] = config["x_dim"]
        spec["out_dim"] = 2*config["y_dim"]

    def loss(self, logits, y_batch, **kwargs):
        means, alphas = self.split_logits(logits)
        squared_diffs = torch.pow((y_batch - means), 2)
        loss_components = torch.sum((alphas + torch.exp(-1.0 * alphas)*squared_diffs),
                dim=1)
        loss = 0.5*torch.mean(loss_components)
        return loss

    # Split logits into mean and alpha = log(variance)
    def split_logits(self, logits):
        means = logits[:, :self.y_dim]
        alphas = logits[:, self.y_dim:] # alpha = log(sigma^2)
        return means, alphas

    @torch.no_grad()
    def get_pdf(self, x, **kwargs):
        x_batch = self.prepare_x_pdf(x)
        logits = self.net(x_batch)
        means, alphas = self.split_logits(logits)
        return utils.get_gaussian_pdf(means, torch.exp(0.5*alphas))

    @torch.no_grad()
    def sample(self, xs, batch_size=None, **kwargs):
        logits = self.sample_batched(xs, batch_size)
        means, alphas = self.split_logits(logits)
        noise = torch.randn_like(means)
        stds = torch.exp(0.5*alphas)
        samples = means + noise*stds
        return samples

    @torch.no_grad()
    def eval(self, dataset, config, use_best_kernel_scale=False):
        x = dataset.x.to(self.device)
        y_true = dataset.y.to(self.device)
        logits = self.net(x)
        means, alphas = self.split_logits(logits)

        # Compute log-likelihood
        # Loss is negative log-likelihood - const.
        loss = self.loss(logits, y_true)
        ll = -1.0 * loss - 0.5*self.y_dim*math.log(2.0*math.pi)

        # Compute MAE
        abs_diff = torch.abs(y_true - means)
        mae = torch.mean(torch.sum(abs_diff, dim=1))
        return {"ll": ll, "mae": mae,}

