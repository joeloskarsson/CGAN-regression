import torch
import torch.utils.data as torch_data
import torch.nn as nn
import torch.distributions as tdists
import math
import numpy as np

from tabular_dataset import TabularDataset
import constants
from models.nn import NN
import utils

# Neural network based Energy model, Deep Conditional Target Density (DCTD)
# from Gustafsson et al.
class DCTD(NN):
    def __init__(self, config):
        super().__init__(config)
        # Amount of samples used in importance sampling
        self.imp_samples = config["imp_samples"]

        # Gaussians making up proposal distribution
        self.y_dim = config["y_dim"]
        n_proposals = len(config["proposal_scales"])
        means = torch.zeros((n_proposals, config["y_dim"]), device=config["device"])
        scales = torch.tensor(config["proposal_scales"], device=config["device"])
        eye = torch.eye(config["y_dim"], device=config["device"])
        covs = scales.view(n_proposals, 1, 1) * eye.view(
                1, config["y_dim"], config["y_dim"])

        # Produces samples of shape (n_proposals, y_dim)
        self.proposals = tdists.multivariate_normal.MultivariateNormal(means, covs)

        # Parameters for mode finding procedure to get
        # center for proposal distribution at test time
        self.mode_find_lr = config["mode_find_lr"]
        self.mode_find_steps = config["mode_find_steps"]

    def compute_dims(self, config, spec):
        # Simplest case, takes just x in
        spec["in_dim"] = config["x_dim"] + config["y_dim"]
        spec["out_dim"] = 1
        spec["x_dim"] = config["x_dim"]
        spec["other_dim"] = config["y_dim"]

    # Only net output for true (x,y) handled here, importance
    # sampling y:s handled in loss function directly
    def process_net_input(self, x_batch, y_batch, **kwarg):
        return torch.cat((x_batch, y_batch), dim=1)

    # Sample from zero-centered proposal
    def sample_proposal(self, n_batch):
        proposal_samples = self.proposals.sample((n_batch, self.imp_samples))
        # Shape (n_batch, n_imp_samples, n_proposals, dim_y)

        n_proposals = proposal_samples.shape[2]
        # Index of component to take sample from
        comp_index = torch.randint(low=0, high=n_proposals,
                size=(n_batch, self.imp_samples), device=self.device)

        indexes = comp_index.unsqueeze(2).unsqueeze(3).repeat(1,1,1,self.y_dim)
        samples = torch.gather(proposal_samples, dim=2,
                index=indexes).squeeze(2)
        # Shape (n_batch, n_imp_samples, dim_y)

        return samples

    def estimate_log_z(self, x_batch, proposal_means, return_weights=False):
        n_batch = x_batch.shape[0]
        # Sample from proposal distribution
        prop_samples = self.sample_proposal(n_batch)
        # Shape (n_batch, n_imp_samples, dim_y)

        # Compute importance weights
        component_log_pds = self.proposals.log_prob(prop_samples.unsqueeze(2))
        # Shape (n_batch, n_imp_samples, n_components)
        n_comp = component_log_pds.shape[2]
        # Compute using logsumexp for numerical stability
        proposal_log_pd = torch.logsumexp(component_log_pds, dim=2) - math.log(n_comp)
        # Shape (n_batch, n_imp_samples)

        # Move importance samples to reflect mean of actual proposal distribution
        prop_samples_moved = prop_samples + proposal_means.unsqueeze(1)

        imp_nn_inputs = torch.cat((
            torch.repeat_interleave(x_batch, self.imp_samples, dim=0),
            prop_samples_moved.reshape(n_batch*self.imp_samples, -1) # -1 will be dim_y
            ), dim=1)
        imp_nn_out = self.net(imp_nn_inputs)
        # Shape (n_batch*n_imp_samples, 1)
        unnorm_model_log_pd = imp_nn_out.reshape(n_batch, self.imp_samples)
        # Shape (n_batch,n_imp_samples)

        # Work in log-space, logsumexp for numerical stability
        log_pd_diff = unnorm_model_log_pd - proposal_log_pd
        log_s = torch.logsumexp(log_pd_diff, dim=1)
        log_z = log_s - math.log(self.imp_samples)
        # Shape (n_batch)

        if return_weights:
            # Compute and return importance weights and samples
            imp_weights = torch.exp(log_pd_diff - log_s.unsqueeze(1))
            return log_z, imp_weights, prop_samples_moved
        else:
            return log_z

    def loss(self, logits, y_batch, x_batch, **kwargs):
        # Logits are unnormalized pdf at (x,y), f(x,y) in DCTD paper
        log_z = self.estimate_log_z(x_batch, y_batch) # return shape (n_batch)

        loss = torch.mean(log_z - logits.squeeze())
        return loss

    @torch.no_grad()
    def get_pdf(self, x, n_samples=100, **kwargs):
        if type(x) == torch.Tensor:
            x = x.to(self.device)
        else:
            x = torch.tensor([x], device=self.device)

        x = x.repeat(1, 1) # Make sure x has correct dimensions

        proposal_means = self.find_modes(x)
        log_z = self.estimate_log_z(x, proposal_means).squeeze()
        def pdf(y):
            # Note that y is assumed to be 1-dimensional
            net_input = torch.cat((x, torch.tensor([[y]], device=self.device)), dim=1)
            net_out = self.net(net_input).squeeze()

            pd = torch.exp(net_out - log_z)
            return pd.to("cpu").item()

        return pdf

    # Find modes of model distribution (useful for selecting
    # proposal distributions in importance sampling)
    # Requires gradients for optimization process
    @torch.enable_grad()
    def find_modes(self, xs):
        # Initialize optimization at y=0
        y_param = nn.Parameter(torch.zeros((xs.shape[0], self.y_dim),
            device=self.device))
        opt = torch.optim.Adam([y_param], lr=self.mode_find_lr)

        for _ in range(self.mode_find_steps):
            opt.zero_grad()

            net_in = torch.cat((xs, y_param), dim=1)
            pd = self.net(net_in)
            loss = torch.sum(-pd)

            loss.backward()
            opt.step()

        modes = y_param.data
        return modes

    @torch.no_grad()
    def sample(self, xs, batch_size=None, **kwargs):
        xs = xs.to(self.device)

        # Fix since x is unnecessarily repeated
        unique_xs, counts = torch.unique_consecutive(xs, return_counts=True, dim=0)
        n_samples = counts[0].item() # Assume all counts the same

        assert n_samples < self.imp_samples, (
          """Trying to draw more or same amount of samples from DCTD model as is used
          for importance sampling. Increase amount of importance samples used.""")

        # Batch over x:s
        if batch_size:
            batches = torch.split(unique_xs, batch_size, dim=0)
        else:
            batches = (unique_xs,)

        sample_list = []
        for x_batch in batches:
            proposal_means = self.find_modes(x_batch)

            _, imp_weights, imp_samples = self.estimate_log_z(x_batch,
                proposal_means, return_weights=True)
            # imp_samples shape: (n_batch, n_imp_samples, dim_y)

            # Sample index from categorical distribution parametrized
            # by importance weights
            sample_indexes = torch.multinomial(imp_weights, n_samples, replacement=True)
            # shape: (n_batch, n_samples)

            indexes = sample_indexes.unsqueeze(2).repeat(1,1,self.y_dim)
            sample_batch = torch.gather(imp_samples, dim=1,
                index=indexes)
            # Shape (n_batch, n_samples, dim_y)
            sample_batch_flat = torch.flatten(sample_batch, start_dim=0, end_dim=1)
            # Shape (n_batch*n_samples, dim_y)

            sample_list.append(sample_batch_flat)

        samples = torch.cat(sample_list, dim=0)
        return samples

    @torch.no_grad()
    def eval(self, dataset, config, use_best_kernel_scale=False):
        tab_dataset = TabularDataset(dataset)
        loader = torch_data.DataLoader(tab_dataset,
                batch_size=config["eval_batch_size"], shuffle=False,
                num_workers=constants.LOAD_WORKERS)

        total_ll = 0.0
        total_mae = 0.0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            logits = self.net(self.process_net_input(x_batch, y_batch))
            proposal_means = self.find_modes(x_batch)
            log_z, imp_weights, batch_samples = self.estimate_log_z(x_batch,
                    proposal_means, return_weights=True)

            batch_ll = torch.sum(-log_z + logits.squeeze())
            total_ll += batch_ll.item()

            sample_maes = torch.sum(torch.abs(y_batch.unsqueeze(1) - batch_samples),
                    dim=2)
            batch_mae = torch.sum(torch.sum(imp_weights*sample_maes, dim=1))
            total_mae += batch_mae.item()

        data_len = len(tab_dataset)
        ll = total_ll / data_len
        mae = total_mae / data_len

        # Not entirely sure if MAE is correct here, so it should probably not be trusted
        # Use MAE from KDE-evaluation instead
        return {"ll": ll, "mae": mae,}

