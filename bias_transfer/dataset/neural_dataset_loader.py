import numpy as np
import torch

from nnfabrik import builder


def neural_dataset_loader(seed, **config):
    config.pop('comment', None)
    torch.manual_seed(seed)
    np.random.seed(seed)
    dataset_fn = 'nnvision.datasets.monkey_static_loader'
    data_loaders = builder.get_data(dataset_fn, config)
    return data_loaders
