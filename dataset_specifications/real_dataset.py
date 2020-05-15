from dataset_specifications.dataset import Dataset

# Abstract class for datasets based on real data
class RealDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.synthetic = False
        self.requires_path = True

        self.val_percent = 0.15 # Percentage of real datasets to use for validation
        self.test_percent = 0.15 # Percentage of real datasets to use for testing

    def get_pdf(self, x):
        raise NotImplementedError("Pdf for real world dataset not implemented")

    def preprocess(self, file_path):
        raise NotImplementedError("Preprocessing method needs to be implemented")

