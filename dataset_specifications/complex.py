import numpy as np

import utils
from dataset_specifications.dataset import Dataset

class ComplexSet(Dataset):
    def __init__(self):
        super().__init__()
        self.name = "complex"

        self.support = (-2., 2.)

    def std(self, xs):
        std = -0.5*xs + 0.1
        std[xs >= 0.] = 0.1
        return std

    def sample(self, n):
        xs = np.random.uniform(low=-2., high=2., size=n)
        noise = np.random.normal(loc=0., scale=self.std(xs), size=n)

        component = np.random.randint(2, size=n) # p(0) = p(1) = 0.5
        means = (1-component)*(0.5*xs) + component*(-0.5*xs)
        means[xs < 0.] = 0.
        ys = means + noise

        return np.stack((xs, ys), axis=1)

    def get_pdf(self, x):
        if x < 0:
            return utils.get_gaussian_pdf(mean=0., std_dev=-0.5*x + 0.1)
        else:
            g0 = utils.get_gaussian_pdf(mean=0.5*x, std_dev=0.1)
            g1 = utils.get_gaussian_pdf(mean=-0.5*x, std_dev=0.1)

            return (lambda y: 0.5*(g0(y) + g1(y)))

