import numpy as np
import torch

from dataset_specifications.dataset import Dataset

class BimodalSet(Dataset):
    def __init__(self):
        super().__init__()
        self.name = "bimodal"

    def f0(self, x):
        return 0.5*x + 2.5

    def f1(self, x):
        return -0.5*x

    def get_support(self, x):
        return (self.f1(x)-0.6, self.f0(x)+0.6)

    def sample(self, n):
        xs = np.random.uniform(low=-1., high=1., size=n)
        noise = np.random.uniform(low=-0.5, high=0.5, size=n)
        component = np.random.randint(2, size=n) # p(0) = p(1) = 0.5
        ys = (1-component)*self.f0(xs) + component*self.f1(xs) + noise

        return np.stack((xs, ys), axis=1)

    def get_pdf(self, x):
        l0 = [self.f0(x)-0.5, self.f0(x)+0.5]
        l1 = [self.f1(x)-0.5, self.f1(x)+0.5]

        def pdf(y):
            if ((l0[0] <= y) and (y <= l0[1])) or ((l1[0] <= y) and (y <= l1[1])):
                return 0.5
            else:
                return 0.

        return pdf

    # Evaluation metric: Fraction of points in distribution support
    def evaluation_metric(self, xs, samples):
        # Data has 1 y-dim and 1 x-dim, so squeeze
        samples = samples.squeeze(dim=2)

        # Get mean lines of components
        m0 = self.f0(xs)
        m1 = self.f1(xs)

        # Everything within distance <= 0.5 of a mean line is in support
        dist0 = torch.abs(samples-m0)
        dist1 = torch.abs(samples-m1)
        in_support = (dist0 <= 0.5) | (dist1 <= 0.5)
        return torch.mean(in_support.to(float)).item()

