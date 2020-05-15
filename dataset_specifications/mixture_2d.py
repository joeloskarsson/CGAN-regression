import numpy as np
import math

from dataset_specifications.dataset import Dataset

class Mixture2DSet(Dataset):
    def __init__(self):
        super().__init__()
        self.name = "mixture_2d"

        # Mixture of 6 2D isotropic gaussians, laid out along a unit circle
        # Not conditional, so all x:s are just 0

        self.y_dim = 2
        self.std = 0.15

    def sample_ys(self, xs):
        n = xs.shape[0]
        components = np.random.randint(6, size=n) # uniform
        angles = (math.pi/3.0) * components # Angles to centers
        means = np.stack((np.cos(angles), np.sin(angles)), axis=1)

        noise = np.random.randn(n, 2) # samples form 2d gaussian

        ys = means + self.std*noise
        return ys

    def sample(self, n):
        xs = np.zeros((n,1))
        ys = self.sample_ys(xs)
        return np.concatenate((xs, ys), axis=1)


