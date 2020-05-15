import numpy as np

import utils
from dataset_specifications.dataset import Dataset

class ExponentialSet(Dataset):
    def __init__(self):
        super().__init__()
        self.name = "exponential"

    def f(self, x):
        return -0.5*np.power(x, 2)

    def exp_mean(self, x):
        return 0.3*(1. - np.abs(x))

    def sample(self, n):
        xs = np.random.uniform(low=-1., high=1., size=n)
        noise = np.random.exponential(scale=self.exp_mean(xs), size=n)
        ys = self.f(xs) + noise

        return np.stack((xs, ys), axis=1)

    def get_support(self, x):
        f = self.f(x)
        return (f-0.1, f+3*self.exp_mean(x))

    def get_pdf(self, x):
        f = self.f(x)

        exp_pdf = utils.get_exponential_pdf(mean=self.exp_mean(x))
        def pdf(x):
            if x >= f:
                return exp_pdf(x-f) # Distance above f-line
            else:
                return 0.
        return pdf

