import numpy as np
import math

from dataset_specifications.dataset import Dataset

class SwirlsSet(Dataset):
    def __init__(self):
        super().__init__()
        self.name = "swirls"

        self.n_samples = {
            "train": 2000,
            "val": 1000,
            "test": 1000,
        }

        self.y_dim = 2

        # 2D heteroskedastic Gaussian mixture model with 2 components

    def sample_ys(self, xs):
        n = xs.shape[0]
        components = np.random.randint(2, size=n) # uniform 0,1
        angles = math.pi*components + (math.pi/2.)*xs[:,0] # Angles to centers
        means = np.stack((np.cos(angles), np.sin(angles)), axis=1)

        noise = np.random.randn(n, 2) # samples form 2d gaussian

        std = 0.3 - 0.2*np.abs(xs-1.)
        ys = means + std*noise
        return ys

    def sample(self, n):
        xs = np.random.uniform(low=0., high=2., size=(n,1))
        ys = self.sample_ys(xs)
        return np.concatenate((xs, ys), axis=1)


