import numpy as np

from dataset_specifications.dataset import Dataset

class ButterflySet(Dataset):
    def __init__(self):
        super().__init__()
        self.name = "butterfly"
        self.support = (0.,4.)

    def sample(self, n):
        xs = np.random.uniform(low=-2., high=2., size=n)
        g_noise = np.random.normal(loc=0., scale=1., size=n)
        ys = np.log(1. + np.exp((-1.)*xs*g_noise))
        return np.stack((xs, ys), axis=1)

