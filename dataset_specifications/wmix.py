import numpy as np
import torch
import torch
from functools import partial

from dataset_specifications.dataset import Dataset

class WmixSet(Dataset):
    def __init__(self, n_comp):
        super().__init__()

        self.n_samples = {
            "train": 50000,
            "val": 5000,
            "test": 5000,
        }

        self.n_comp = n_comp

        self.x_dim = 3*n_comp
        self.y_dim = 1

        self.name = "wmix{}".format(self.x_dim)
        self.support = (0.,10.)

        # Mixture of n_comp Weibull distributions
        # x is parameters of mixture, length 3*n*comp
        # x[1:n_comp] are component offsets (added onto samples)
        # x[n_comp:2n_comp] are component weights (unnormalized)
        # x[2n_comp:3n_comp] are shape modifiers for the Weibull components
        # The shape parameter of each component is 1 + shape_modifier

    def sample_ys(self, xs):
        comp_offsets= xs[:,:self.n_comp]
        weights = xs[:,self.n_comp:2*self.n_comp]
        comp_shape_mods = xs[:,2*self.n_comp:]

        # Component probabilities (normalized weights)
        comp_probs = weights / np.sum(weights, axis=1, keepdims=True)
        comp_i = np.apply_along_axis(
            (lambda probs: np.random.choice(self.n_comp, size=1, p=probs)),
                1, comp_probs) # Axis should be 1, see numpy docs
        # Shape (n, 1)

        # Get values for chosen component (component with sampled index)
        offsets = np.take_along_axis(comp_offsets, comp_i, 1)
        shape_mods = np.take_along_axis(comp_shape_mods, comp_i, 1)
        shape_params = 1. + shape_mods

        w_samples = np.random.weibull(shape_params)
        # Shape (n, 1)
        ys = w_samples + 5.*offsets

        return ys

    def sample(self, n):
        # xs is from Uniform(0,1)^(dim(x))
        xs = np.random.uniform(low=0.0, high=1.0, size=(n, self.x_dim))
        ys = self.sample_ys(xs)
        return np.concatenate((xs, ys), axis=1)

    def get_pdf(self, x):
        # Perform pdf computation in pytorch
        if type(x) == torch.Tensor:
            x = x.to("cpu")
        else:
            x = torch.tensor(x)

        comp_offsets = x[:self.n_comp]
        weights = x[self.n_comp:2*self.n_comp]
        comp_shape_mods = x[2*self.n_comp:]

        comp_probs = weights / torch.sum(weights)

        shapes = 1.0 + comp_shape_mods
        scales = torch.ones_like(shapes)

        w_dists = torch.distributions.weibull.Weibull(scales, shapes)

        def pdf(y):
            no_offsets = y - 5.0*comp_offsets
            positive = (no_offsets >= 0.0)
            # Only use probability density for positive samples (within support)
            # Pytorch gives positive density even outside support for some reason

            log_probs = w_dists.log_prob(no_offsets)

            filtered_probs = torch.exp(log_probs[positive])
            pd = torch.sum(filtered_probs * comp_probs[positive])
            return pd

        return pdf

WMIX_SET_DICT = {
        "wmix{}".format(3*n_comp):
        partial(WmixSet, n_comp)
    for n_comp in [1,2,3,5]
}

