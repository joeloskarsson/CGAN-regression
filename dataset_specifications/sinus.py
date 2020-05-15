import numpy as np

import utils
from dataset_specifications.dataset import Dataset

class Sinus(Dataset):
    def __init__(self):
        super().__init__()
        self.name = "sinus"

       # Sinus dataset from Aggarwal et. al.

    def get_support(self, x):
        sin = np.sin(x)
        return (sin-2, sin+2)

    def sample(self, n):
        xs = np.random.uniform(low=-4., high=4., size=n)
        # Note mistake in original paper, noise std_dev should be 0.4 (see their code)
        noise = np.random.normal(loc=0., scale=0.4, size=n)
        ys = np.sin(xs) + noise

        return np.stack((xs, ys), axis=1)

    def get_pdf(self, x):
        return utils.get_gaussian_pdf(np.sin(x), 0.4)

