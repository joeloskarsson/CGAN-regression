import numpy as np

from dataset_specifications.real_dataset import RealDataset

class MicroWaveSet(RealDataset):
    def __init__(self):
        super().__init__()
        self.name = "microwave"

        self.support = (-6., 7.2)

        # Microwave radiation dataset from Wilkinson Microwave Anisotropy Probe
        # Original paper: https://iopscience.iop.org/article/10.1086/377253
        # Also used in: https://arxiv.org/abs/0909.5194
        #
        # Dataset can be downloaded from:
        # https://lambda.gsfc.nasa.gov/data/map/powspec/map_comb_tt_powspec_yr1_v1p1.txt
        #
        # x is multipole moment
        # y is TT power spectrum

    def preprocess(self, file_path):
        loaded_data = np.loadtxt(file_path)
        dataset = loaded_data[:,:2] # x and y both 1D

        # Normalize x to [0,1]
        dataset[:,0] = (dataset[:,0] - 2.0)/898.0

        # Standardize y
        dataset[:,1] = (dataset[:,1] - np.mean(dataset[:,1]))/np.std(dataset[:,1])

        return dataset


