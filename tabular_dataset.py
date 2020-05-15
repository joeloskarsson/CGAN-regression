from torch.utils.data import Dataset
import torch

# Pytorch dataset for tabular data (x and y are simply 2D tensors)
class TabularDataset(Dataset):
    def __init__(self, labelled_data):
        super(TabularDataset, self).__init__()
        self.xs = labelled_data.x
        self.ys = labelled_data.y

    def __len__(self):
        return self.xs.shape[0]

    def __getitem__(self, idx):
        x_row = self.xs[idx, :]
        y_row = self.ys[idx, :]

        return x_row, y_row

