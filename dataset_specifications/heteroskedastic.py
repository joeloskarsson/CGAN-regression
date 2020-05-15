import numpy as np
import torch

import utils
from dataset_specifications.dataset import Dataset

class HeteroskedasticSet(Dataset):
    def __init__(self):
        super().__init__()
        self.name = "heteroskedastic"

    def mean_func(self, x):

        return np.power(x,2) + 0.5

    def std_dev(self, x):
        return np.power(np.sin(np.pi*x), 2) + 0.01

    def get_support(self, x):
        mean = self.mean_func(x)
        std_dev = self.std_dev(x)
        return (mean-2*std_dev, mean+2*std_dev)

    def sample(self, n):
        xs = np.random.uniform(low=0., high=1., size=n)
        noise = np.random.normal(loc=0., scale=self.std_dev(xs), size=n)
        ys = self.mean_func(xs) + noise

        return np.stack((xs, ys), axis=1)

    def get_pdf(self, x):
        return utils.get_gaussian_pdf(self.mean_func(x), self.std_dev(x))

    # Evaluation metric: Difference in std-dev
    def evaluation_metric(self, xs, samples):
        # Data has 1 y-dim and 1 x-dim, so squeeze
        samples = samples.squeeze(dim=2)
        xs = xs.squeeze(dim=1)

        true_std = torch.pow(torch.sin(np.pi*xs), 2) + 0.01
        est_std = torch.std(samples, dim=1)

        std_diff = torch.abs(true_std - est_std)
        return torch.mean(std_diff).item()

