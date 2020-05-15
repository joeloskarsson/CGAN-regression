import numpy as np

import utils
from dataset_specifications.dataset import Dataset

class LaplaceSet(Dataset):
    def __init__(self):
        super().__init__()
        self.name = "laplace"
        self.scale=0.5

    def mean_func(self, x):
        return 0.5*x

    def sample(self, n):
        xs = np.random.normal(loc=0., scale=1., size=n) # Note that x here is Gaussian
        noise = np.random.laplace(loc=0., scale=self.scale, size=n)
        ys = self.mean_func(xs) + noise

        return np.stack((xs, ys), axis=1)

    def get_support(self, x):
        mean = self.mean_func(x)
        return (mean-3*self.scale, mean+3*self.scale)

    def get_pdf(self, x):
        return utils.get_laplace_pdf(mean=self.mean_func(x), scale=self.scale)

