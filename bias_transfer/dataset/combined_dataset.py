from collections import namedtuple
from torch.utils.data import Dataset


JoinedDataset = namedtuple("JoinedDataset", "sample_datasets target_datasets")


class CombinedDataset(Dataset):
    """
    TODO documentation
    """

    def __init__(self, *parallel_datasets):
        self.parallel_datasets = parallel_datasets

    def __getitem__(self, index):
        ds_index = index % len(
            self.parallel_datasets
        )  # this samples equally from all parallel datasets
        # TODO implement flexible sampling rates
        joined_dataset = self.parallel_datasets[ds_index]
        in_ds_index = (index // len(self.parallel_datasets)) % len(
            joined_dataset.sample_datasets[0]
        )
        samples = []
        for sample_ds in joined_dataset.sample_datasets:
            s = sample_ds[in_ds_index]
            if isinstance(s, tuple):
                s = s[0]
            samples.append(s)
        targets = []
        for target_ds in joined_dataset.target_datasets:
            t = target_ds[in_ds_index]
            if isinstance(t, tuple):
                t = t[-1]
            targets.append(t)
        return_val = samples + targets
        return return_val

    def __len__(self):
        return max(
            [len(ds.sample_datasets[0]) for ds in self.parallel_datasets]
        )  # all sample_datasets have same length
