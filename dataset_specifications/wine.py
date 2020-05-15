import numpy as np
from sklearn import preprocessing as skpp

from dataset_specifications.real_dataset import RealDataset

# The wine dataset is not very useful in the probabilistic setting.
# Since y takes on a discrete set of values it is very much possible to achieve infinite
# log-likelihood for multiple models.
class WineSet(RealDataset):
    def __init__(self):
        super().__init__()
        self.name = "wine"

        self.x_dim = 11
        self.support = (0.,1.)

        self.val_percent = 0.20
        self.test_percent = 0.20

        # (White) Wine dataset
        # See: https://archive.ics.uci.edu/ml/datasets/Wine+Quality

    def preprocess(self, file_path):
        # delimiter is ; , top row is comment
        loaded_data = np.loadtxt(file_path, delimiter=";", skiprows=1)

        xs = loaded_data[:,:-1]
        # Standardize x
        loaded_data[:, :-1] = skpp.StandardScaler().fit(xs).transform(xs)

        # Tranform y to [0,1], (in reality 0.3 - 0.9 since no wine scores <3 or >9)
        loaded_data[:,-1] *= 0.1

        return loaded_data

