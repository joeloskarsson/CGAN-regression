import torch.nn as nn
import torch

from models.nn import NN
import utils

# Neural network regression model based on iid additive gaussian noise
# of constant variance
# Equal to p(y|x) = N(Y|NN(x), s^2), for constant s^2 (estimated with ML)
class NNRegressor(NN):
    def __init__(self, config):
        super().__init__(config)

        self.mse_loss = nn.MSELoss()

    def loss(self, logits, y_batch, **kwargs):
        return self.mse_loss(logits, y_batch)

    def register_buffers(self, config):
        # Standard deviation of noise
        self.std = torch.ones(1,config["y_dim"], device=self.device)
        # Make noise std network parameter to allow for saving and loading of it
        self.net.register_buffer("std", self.std)

    # Determine std-dev of noise from training data (ML)
    # See eq. 3.21. in Bishop etc.
    # Determined for each y-dimensions separately
    @torch.no_grad()
    def learn_std(self, dataset):
        x = dataset.x.to(self.device)
        y_true = dataset.y.to(self.device)
        y_pred = self.net(x)
        squared_diffs = torch.pow(y_true - y_pred, 2)
        self.std[0] = torch.sqrt(torch.mean(squared_diffs, dim=0))

    # Override train to change val_func to always also learn std before validation
    def train(self, train_set, config, val_func=None):
        if val_func:
            def new_val_func(self, epoch_i):
                self.learn_std(train_set)
                return val_func(self, epoch_i)
        else:
            new_val_func = None

        super().train(train_set, config, new_val_func)

    @torch.no_grad()
    def get_pdf(self, x, **kwargs):
        x_batch = self.prepare_x_pdf(x)
        mean = self.net(x_batch)
        return utils.get_gaussian_pdf(mean, self.std[0])

    @torch.no_grad()
    def sample(self, xs, batch_size=None, **kwargs):
        means = self.sample_batched(xs, batch_size)
        noise = torch.randn_like(means)
        samples = means + noise*self.std[0]
        return samples

    @torch.no_grad()
    def eval(self, dataset, config, use_best_kernel_scale=False):
        x = dataset.x.to(self.device)
        y_true = dataset.y.to(self.device)
        y_pred = self.net(x)

        # Compute log-likelihood
        # Note that N(y|m(x),s) = N(y-m(x)|0,s)
        covariance = torch.diag_embed(torch.pow(self.std, 2))
        dist = torch.distributions.multivariate_normal.MultivariateNormal(
                torch.zeros(config["y_dim"], device=self.device),
                covariance)
        lls = dist.log_prob(y_true - y_pred)
        ll = torch.mean(lls)

        # Compute MAE
        abs_diff = torch.abs(y_true - y_pred)
        mae = torch.mean(torch.sum(abs_diff, dim=1))
        return {"ll": ll, "mae": mae,}

    def post_training(self, train_set, config):
        self.learn_std(train_set)

