import numpy as np
import torch
import collections
import os

import constants

LabelledData = collections.namedtuple("LabelledData",
        ["x", "y", "spec"], defaults=(None,)*3)

class Dataset():
    def __init__(self):
        self.n_samples = {
            "train": 1000,
            "val": 1000,
            "test": 1000,
        }
        self.name = "dataset_name"
        self.synthetic = True # If dataset is synthetic
        self.requires_path = False # If a file path is required in preprocessing

        self.support = (-1.,1.) # Meaningful area of interest of x (for plots)

        self.x_dim = 1
        self.y_dim = 1

    def load(self, device):
        dataset_dir = os.path.join(constants.DATASET_PATH, self.name)
        assert os.path.exists(dataset_dir), (
            "Dataset folder '{}' does not exist".format(dataset_dir))

        splits = {}
        for split in ["train", "val", "test"]:
            data_path = os.path.join(dataset_dir, "{}.csv".format(split))
            data = np.genfromtxt(data_path, delimiter=",")

            # Always have at least 2 dimensions
            if len(data.shape) == 1:
                data = np.expand_dims(data, axis=1)

            # Always put datasets in CPU memory
            torch_data = torch.tensor(data, device="cpu").float()
            splits[split] = LabelledData(
                    x=torch_data[:,:self.x_dim], y=torch_data[:,self.x_dim:], spec=self)

        return splits["train"], splits["val"], splits["test"]

    # Not actually support, but rather area of interest of pdf
    def get_support(self, x):
        return self.support

    def sample(self, n):
        raise NotImplementedError("Can not sample from dataset")

    def get_pdf(self, x):
        raise NotImplementedError("No pdf implemented")

    def evaluation_metric(self, xs, samples):
        # Evaluation metric specific to the dataset
        # Return 'None' for no metric tied to the dataset
        return None

