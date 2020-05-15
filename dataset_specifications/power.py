import numpy as np
import pandas as pd

from dataset_specifications.real_dataset import RealDataset
from sklearn import preprocessing as skpp

class PowerSet(RealDataset):
    def __init__(self):
        super().__init__()
        self.name = "power"

        self.x_dim = 4
        self.support = (-2,2.5)

        # Power Plant dataset, regress hourly electrical output
        # from physical measurements
        # Can be downloaded from:
        # https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant
        # (use xlsx file)

    def preprocess(self, file_path):
        ws = pd.read_excel(file_path, sheet_name="Sheet1")
        loaded_data = np.array(ws)

        # Standardize all data
        loaded_data = skpp.StandardScaler().fit(loaded_data).transform(loaded_data)

        return loaded_data


