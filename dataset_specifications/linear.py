import numpy as np

import utils
from dataset_specifications.dataset import Dataset

class Linear(Dataset):
    def __init__(self):
        super().__init__()
        self.name = "linear"

       # Linear dataset from Aggarwal et. al.

    def get_support(self, x):
        return (x-3, x+9)

    def sample(self, n):
        xs = np.random.normal(loc=4., scale=3., size=n)
        noise = np.random.normal(loc=3., scale=3., size=n)
        ys = xs + noise

        return np.stack((xs, ys), axis=1)

    def get_pdf(self, x):
        return utils.get_gaussian_pdf(x+3, 3)

