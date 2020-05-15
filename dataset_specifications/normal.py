import numpy as np

import utils
from dataset_specifications.dataset import Dataset

class NormalSet(Dataset):
    def __init__(self):
        super().__init__()
        self.name = "normal"
        self.support = (-2.,2.)

        # Samples from a standard gaussian
        # Not conditional, so all x:s are just 0

    def sample(self, n):
        ys = np.random.randn(n)
        xs = np.zeros(n)
        return np.stack((xs, ys), axis=1)

    def get_pdf(self, x):
        return utils.get_gaussian_pdf(0, 1)

