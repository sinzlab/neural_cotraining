import numpy as np
from torch.utils.data import Dataset


class ManyDatasetsInOne(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        data1 = self.datasets[0][idx]
        data2 = self.datasets[1][idx]
        images = [np.asarray(data1[0]), np.asarray(data2[0])]
        labels = data1[1]
        return images, labels
