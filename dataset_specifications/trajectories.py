import numpy as np
import math
from functools import partial

from dataset_specifications.dataset import Dataset

class TrajectoriesSet(Dataset):
    def __init__(self, dim_y):
        super().__init__()
        self.name = "trajectories{}".format(dim_y)

        self.n_samples = {
            "train": 20000,
            "val": 4000,
            "test": 4000,
        }

        assert (dim_y % 2) == 0, "Trajectories y-dimension must be even"
        self.steps = dim_y // 2

        self.x_dim = 5
        self.y_dim = 2*self.steps

        # Generates 2D-trajectories according to a stochastic process
        # x-vector are parameters of the process in the order:
        # x_1, start x-position
        # x_2, start y-position
        # x_3, step length
        # x_4, initial direction (angle as multiple of pi)
        # x_5, turn range between steps

    def sample_ys(self, xs):
        n = xs.shape[0]
        x0 = xs[:,(0,1)]
        l = xs[:,(2,)]
        theta_init = xs[:,(3,)]
        max_theta = xs[:,(4,)]

        # Roll out trajectories (ys)
        standard_theta = np.random.uniform(low=-math.pi, high=math.pi,
                size=(n,self.steps)) # Sample angle uniformly on [-pi, pi]
        rel_theta = max_theta*standard_theta # Scale to get in conditional range
        abs_theta = np.cumsum(rel_theta, axis=1) + theta_init

        # these two have shape (n,steps,2)
        delta_steps = np.expand_dims(l, axis=2) *\
            np.stack((np.cos(abs_theta), np.sin(abs_theta)), axis=2)
        positions = np.cumsum(delta_steps, axis=1) + np.expand_dims(x0, axis=1)

        ys = positions.reshape(n,self.y_dim) # Flatten to (x00,x01, x10, x11, .., )
        return ys

    def sample(self, n):
        # Construct xs
        x0 = np.random.randn(n,2) # n samples from 2-dim isotropic standard Gaussian
        max_step = 1./self.steps
        l = np.random.uniform(low=0.25*max_step, high=max_step, size=(n,1))
        theta_init = np.random.uniform(low=-math.pi, high=math.pi, size=(n,1))
        max_theta = np.random.uniform(low=0.2, high=0.5, size=(n,1))
        xs = np.concatenate((x0, l, theta_init, max_theta), axis=1)

        ys = self.sample_ys(xs)
        return np.concatenate((xs, ys), axis=1)

TRAJECTORIES_SET_DICT = {
        "trajectories{}".format(dim_y):
        partial(TrajectoriesSet, dim_y)
    for dim_y in [4,10,20]
}

