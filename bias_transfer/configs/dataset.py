from .base import BaseConfig
from nnfabrik.main import *
import os
from os import listdir
from os.path import isfile, join

class DatasetConfig(BaseConfig):
    config_name = "dataset"
    table = None
    fn = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = kwargs.pop("batch_size", 128)
        self.update(**kwargs)





class ImageDatasetConfig(DatasetConfig):
    config_name = "dataset"
    table = Dataset()
    fn = "bias_transfer.dataset.img_dataset_loader"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset_cls = kwargs.pop("dataset_cls", "CIFAR10")
        self.apply_augmentation = kwargs.pop("apply_data_augmentation", True)
        self.apply_normalization = kwargs.pop("apply_data_normalization", True)
        self.input_size = kwargs.pop("input_size", 32)
        self.valid_size = kwargs.pop("valid_size", 0.1)
        if self.dataset_cls == "CIFAR100":
            self.train_data_mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
            self.train_data_std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
            self.data_dir = kwargs.pop("data_dir", './data/image_classification/torchvision/')
        elif self.dataset_cls == "CIFAR10":
            self.train_data_mean = (0.49139968, 0.48215841, 0.44653091)
            self.train_data_std = (0.24703223, 0.24348513, 0.26158784)
            self.data_dir = kwargs.pop("data_dir", './data/image_classification/torchvision/')
        elif self.dataset_cls == "TinyImageNet":
            self.train_data_mean = (0.4802, 0.4481, 0.3975)  # (0.485, 0.456, 0.406) mean of full ImageNet
            self.train_data_std = (0.2302, 0.2265, 0.2262)  # (0.229, 0.224, 0.225) std of full ImageNet
            self.data_dir = kwargs.pop("data_dir", './data/image_classification/')
            self.input_size = 64
        else:
            raise NameError()
        self.add_corrupted_test = kwargs.pop("add_corrupted_test", True)
        self.valid_size = kwargs.pop("valid_size", 0.1)
        self.shuffle = kwargs.pop("shuffle", True)
        self.show_sample = kwargs.pop("show_sample", False)
        self.num_workers = kwargs.pop("num_workers", 1)
        self.pin_memory = kwargs.pop("pin_memory", True)
        self.update(**kwargs)


class NeuralDatasetConfig(DatasetConfig):
    config_name = "dataset"
    table = Dataset()
    fn = 'bias_transfer.dataset.neural_dataset_loader'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.train_frac = kwargs.pop("train_frac", 0.8)
        self.dataset = kwargs.pop("dataset", "CSRF19_V1")
        basepath = './data/monkey/toliaslab/{}'.format(self.dataset)
        neuronal_data_path = os.path.join(basepath, 'neuronal_data/')
        self.neuronal_data_files = [neuronal_data_path + f for f in listdir(neuronal_data_path) if
                               isfile(join(neuronal_data_path, f))]
        #self.image_file = os.path.join(basepath, 'images/{}_images.pickle'.format(self.dataset))
        self.image_cache_path = os.path.join(basepath, 'images/individual')

        self.seed = kwargs.pop("seed", 1000)
        self.subsample = kwargs.pop("subsample", 1)
        self.crop = kwargs.pop("crop", 101)
        self.time_bins_sum = kwargs.pop("time_bins_sum", 12)
        self.update(**kwargs)
