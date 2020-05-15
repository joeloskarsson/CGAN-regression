from dataset_specifications.housing import HousingSet

class HouseAgeSet(HousingSet):
    def __init__(self):
        super().__init__()
        self.name = "house_age"

        self.x_dim = 2

        # Modified version of the California house price dataset (housing)
        # Predict house age only from coordinates of house

    def preprocess(self, file_path):
        # Load all data at first
        loaded_data = super().preprocess(file_path)

        # Indexes: 1 - house age, 6 - latitude, 7 longitude
        return loaded_data[:,(6,7,1)]

