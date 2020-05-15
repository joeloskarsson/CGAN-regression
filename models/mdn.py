import torch.nn as nn
import torch
import math
import wandb

from models.nn import NN
import utils

# Mixture density network, predicts parameters of a Gaussian mixture model for each x
# See for example PRML-book by Bishop.
class MDN(NN):
    def __init__(self, config):
        super().__init__(config)

        self.y_dim = config["y_dim"]
        self.n_components = config["mixture_comp"]

    def compute_dims(self, config, spec):
        # n_components + n_components*(2*y_dim) (2 because mean and variance)
        # Order:
        # * mixture components
        # * means (y_1,y_2,..,y_3)*n_components
        # * log-stds (y_1,y_2,..,y_3)*n_components
        spec["out_dim"] = config["mixture_comp"]*(1 + 2*config["y_dim"])
        spec["in_dim"] = config["x_dim"]

    # Split up network output into mixture components, means and variances
    def split_logits(self, logits):
        n_batch = logits.shape[0]
        components = nn.functional.softmax(logits[:, :self.n_components], dim=1)
        n_split = self.n_components * self.y_dim # amount of means and variances
        means = logits[:, self.n_components:(self.n_components + n_split)].reshape(
                n_batch, self.n_components, self.y_dim)
        log_stds = logits[:,
                (self.n_components + n_split):(self.n_components + 2*n_split)
                ].reshape(n_batch, self.n_components, self.y_dim)
        return components, means, log_stds

    # Loss is just mean negative log-likelihood
    def loss(self, logits, y_batch, **kwargs):
        components, means, log_stds = self.split_logits(logits)

        comp_logits = logits[:, :self.n_components]
        log_comps = comp_logits - torch.logsumexp(comp_logits, dim=1, keepdim=True)

        # Must unsqueeze y over mixture dimension
        squared_diffs = torch.pow((y_batch.unsqueeze(1) - means), 2)

        variances = torch.exp(2*log_stds)
        variances = variances.clamp(1e-10,1e10) # Clamp to prevent numerical issues

        log_components = log_comps - 0.5*self.y_dim*math.log(2*math.pi) -\
                torch.sum(log_stds, dim=2) -\
                0.5*torch.sum(squared_diffs/variances, dim=2)

        lls = torch.logsumexp(log_components, dim=1)
        loss = -1*torch.mean(lls)
        return loss

    @torch.no_grad()
    def get_pdf(self, x, **kwargs):
        x_batch = self.prepare_x_pdf(x)
        logits = self.net(x_batch)

        def pdf(y):
            y_batch = torch.tensor([[y]], device=self.device)
            return torch.exp(-1.*self.loss(logits, y_batch))

        return pdf

    @torch.no_grad()
    def sample_from_logits(self, logits):
        components, means, log_stds = self.split_logits(logits)

        # Select which components to sample from, do first so
        # we don't have to sample noise for each gaussian in the mixture
        sample_comps = torch.multinomial(components, 1).squeeze()
        one_hots = nn.functional.one_hot(sample_comps, num_classes=self.n_components)

        sample_means = torch.sum(means*one_hots.unsqueeze(2), dim=1)
        sample_log_stds = torch.sum(log_stds*one_hots.unsqueeze(2), dim=1)

        noise = torch.randn_like(sample_means)
        sample_stds = torch.exp(sample_log_stds)
        samples = sample_means + noise*sample_stds
        return samples

    @torch.no_grad()
    def sample(self, xs, batch_size=None, **kwargs):
        xs = xs.to(self.device)

        if batch_size:
            # Batch sampling process
            batches = torch.split(xs, batch_size, dim=0)

            sample_list = []
            for x_batch in batches:
                logits_batch = self.net(x_batch)
                sample_batch = self.sample_from_logits(logits_batch)
                sample_list.append(sample_batch)

            samples = torch.cat(sample_list, dim=0)
        else:
            logits = self.net(xs)
            samples = self.sample_from_logits(logits)

        return samples

    @torch.no_grad()
    def eval(self, dataset, config, **kwargs):
        x = dataset.x.to(self.device)
        y_true = dataset.y.to(self.device)
        logits = self.net(x)

        # Compute log-likelihood
        # Loss is negative log-likelihood - const.
        loss = self.loss(logits, y_true)
        ll = -1.0 * loss

        # Compute MAE
        components, means, log_stds = self.split_logits(logits)
        if config["log_coefficients"]:
            coeff = components.to("cpu")
            wandb.log({"coefficients": coeff}) # Log mixture coefficients

        pred_means = torch.sum(components.unsqueeze(2) * means, dim=1) # weight means

        abs_diff = torch.abs(y_true - pred_means)
        mae = torch.mean(torch.sum(abs_diff, dim=1))
        return {"ll": ll, "mae": mae,}

