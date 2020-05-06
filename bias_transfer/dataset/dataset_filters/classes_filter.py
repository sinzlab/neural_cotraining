import numpy as np
from .dataset_filter import DatasetFilter


class ClassesFilter(DatasetFilter):
    def __init__(self, config, train_dataset):
        super().__init__(config, train_dataset)
        start, end = config.filter_classes
        self.filtered_classes = train_dataset.classes[start:end]
        self.filtered_classes_idx = list(range(start,end))
        self.percent_start = start / len(train_dataset.classes)
        self.percent_end = end / len(train_dataset.classes)
        self.start = start

    def apply(self, dataset):
        if hasattr(dataset, "data"):
            samples = dataset.data
        else:
            samples = dataset.samples
        filtered_samples, filtered_targets = [], []
        for i, sample in enumerate(samples):
            if dataset.targets[i] in self.filtered_classes_idx:
                filtered_samples.append(sample)
                filtered_targets.append(dataset.targets[i] - self.start)
        if hasattr(dataset, "data"):
            dataset.data = np.stack(filtered_samples)
        else:
            dataset.samples = filtered_samples
        dataset.targets = filtered_targets
        dataset.classes = self.filtered_classes
        if hasattr(dataset, "start"):
            old_length = len(dataset)
            dataset.end = dataset.start + int(self.percent_end * old_length)
            dataset.start += int(self.percent_start * old_length)
        if hasattr(dataset, "target_transform"):
            dataset.target_transform = lambda x: x - self.start
