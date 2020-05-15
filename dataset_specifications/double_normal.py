import numpy as np

import utils
from dataset_specifications.dataset import Dataset

class DoubleNormalSet(Dataset):
    def __init__(self):
        super().__init__()
        self.name = "double_normal"
        self.support = (-4.,4.)

        # Bimodal distribution
        # GMM with pi1 = pi2 = 0.5, two standard normals at -2 and 2
        # Not conditional, so all x:s are just 0

    def sample(self, n):
        gaussians = np.random.randn(n)
        components = np.random.randint(2, size=n) # p(0)=p(1)=0.5
        ys = -2 + (4*components) + gaussians
        xs = np.zeros(n)
        return np.stack((xs, ys), axis=1)

    def get_pdf(self, x):
        pdfs = [utils.get_gaussian_pdf(-2, 1), utils.get_gaussian_pdf(2, 1)]
        return (lambda y: 0.5*pdfs[0](y) + 0.5*pdfs[1](y))

