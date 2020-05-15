import numpy as np

from dataset_specifications.real_dataset import RealDataset
from sklearn import preprocessing as skpp
import sklearn.datasets as skl_ds

class HousingSet(RealDataset):
    def __init__(self):
        super().__init__()
        self.name = "housing"
        self.requires_path = False

        self.x_dim = 8
        self.support = (0.,1.)

        self.val_percent = 0.10
        self.test_percent = 0.10

        # California housing dataset
        # Full dataset is available at http://lib.stat.cmu.edu/datasets/houses.zip
        # Here it is loaded from sci-kit learn librbary, so no file has to
        # be manually downloaded

    def preprocess(self, file_path):
        x,y = skl_ds.fetch_california_housing(return_X_y=True)

        loaded_data = np.concatenate((x, np.expand_dims(y, axis=1)), axis=1)

        # Standardize all data
        loaded_data = skpp.StandardScaler().fit(loaded_data).transform(loaded_data)

        return loaded_data

