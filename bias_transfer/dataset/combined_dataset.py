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
        in_ds_index = (ds_index / len(self.parallel_datasets)) % len(joined_dataset.sample_datasets[0])
        samples = []
        for sample_ds in joined_dataset.sample_datasets:
            samples.append(sample_ds[in_ds_index])
        targets = []
        for target_ds in joined_dataset.target_datasets:
            targets.append(target_ds[in_ds_index])
        return samples, targets

    def __len__(self):
        return max([len(ds.sample_datasets[0]) for ds in self.parallel_datasets])
