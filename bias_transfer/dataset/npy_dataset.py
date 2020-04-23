import os
import numpy as np
import torch
from torchvision.datasets import VisionDataset


class NpyDataset(VisionDataset):
    def __init__(
        self,
        sample_file,
        target_file,
        root,
        start,
        end,
        transforms=None,
        transform=None,
        target_transform=None,
    ):
        super().__init__(root, transforms, transform, target_transform)
        self.sample_file = os.path.join(self.root, sample_file)
        self.target_file = os.path.join(self.root, target_file)
        self.start = start
        self.end = end
        self.sample = None
        self.targets = None

    def load(self):
        if self.sample is None:
            self.sample = np.load(self.sample_file)[self.start : self.end]
        if self.targets is None:
            self.targets = np.load(self.target_file)[self.start : self.end].astype(
                np.int
            )

    def __getitem__(self, index):
        self.load()
        sample = self.sample[index]
        target = self.targets[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return self.end - self.start
