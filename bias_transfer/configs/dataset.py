from .base import BaseConfig
from nnfabrik.main import *


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
        self.apply_grayscale = kwargs.pop("apply_grayscale", False)
        self.apply_noise = kwargs.pop("apply_noise", {})
        self.input_size = kwargs.pop("input_size", 32)
        if self.dataset_cls == "CIFAR100":
            self.train_data_mean = (
                0.5070751592371323,
                0.48654887331495095,
                0.4409178433670343,
            )
            self.train_data_std = (
                0.2673342858792401,
                0.2564384629170883,
                0.27615047132568404,
            )
            self.data_dir = kwargs.pop(
                "data_dir", "./data/image_classification/torchvision/"
            )
            self.num_workers = 1
            self.valid_size = kwargs.pop("valid_size", 0.1)
        elif self.dataset_cls == "CIFAR10":
            self.train_data_mean = (0.49139968, 0.48215841, 0.44653091)
            self.train_data_std = (0.24703223, 0.24348513, 0.26158784)
            self.data_dir = kwargs.pop(
                "data_dir", "./data/image_classification/torchvision/"
            )
            self.num_workers = kwargs.pop("num_workers", 1)
            self.valid_size = kwargs.pop("valid_size", 0.1)
        elif self.dataset_cls == "TinyImageNet":
            if self.apply_grayscale:
                self.train_data_mean = (0.4519,)
                self.train_data_std = (0.2221,)
            else:
                self.train_data_mean = (
                    0.4802,
                    0.4481,
                    0.3975,
                )
                self.train_data_std = (
                    0.2302,
                    0.2265,
                    0.2262,
                )
            self.data_dir = kwargs.pop("data_dir", "./data/image_classification/")
            self.input_size = 64
            self.num_workers = kwargs.pop("num_workers", 1)
            self.valid_size = kwargs.pop("valid_size", 0.1)
        elif self.dataset_cls == "ImageNet":
            if self.apply_grayscale:
                self.train_data_mean = (0.4573,)
                self.train_data_std = (0.2156,)
            else:
                self.train_data_mean = (0.485, 0.456, 0.406)
                self.train_data_std = (0.229, 0.224, 0.225)
            self.data_dir = kwargs.pop("data_dir", "./data/image_classification/")
            self.input_size = 224
            self.num_workers = kwargs.pop("num_workers", 8)
            self.valid_size = kwargs.pop("valid_size", 0.01)
        else:
            raise NameError()
        self.add_corrupted_test = kwargs.pop("add_corrupted_test", True)
        self.add_fly_corrupted_test = kwargs.pop("add_fly_corrupted_test",
                                                 {'jpeg_compression': [1, 2, 3, 4, 5], 'pixelate': [1, 2, 3, 4, 5],
                                                  'impulse_noise': [1, 2, 3, 4, 5], 'defocus_blur': [1, 2, 3, 4, 5],
                                                  'contrast': [1, 2, 3, 4, 5], 'fog': [1, 2, 3, 4, 5],
                                                  'brightness': [1, 2, 3, 4, 5], 'frost': [1, 2, 3, 4, 5],
                                                  'glass_blur': [1, 2, 3, 4, 5], 'shot_noise': [1, 2, 3, 4, 5],
                                                  'motion_blur': [1, 2, 3, 4, 5], 'gaussian_noise': [1, 2, 3, 4, 5],
                                                  'elastic_transform': [1, 2, 3, 4, 5], 'snow': [1, 2, 3, 4, 5],
                                                  'zoom_blur': [1, 2, 3, 4, 5]}
                                                 )
        self.add_stylized_test = kwargs.pop("add_stylized_test", False)
        self.shuffle = kwargs.pop("shuffle", True)
        self.show_sample = kwargs.pop("show_sample", False)
        self.filter_classes = kwargs.pop("filter_classes", None)  # (start,end)
        self.download = kwargs.pop(
            "download", False
        )  # For safety (e.g. to not download ImageNet by accident)
        self.pin_memory = kwargs.pop("pin_memory", True)
        self.update(**kwargs)

    @property
    def filters(self):
        filters = []
        if self.filter_classes:
            filters.append("ClassesFilter")
        return filters


class NeuralDatasetConfig(DatasetConfig):
    config_name = "dataset"
    table = Dataset()
    fn = "bias_transfer.dataset.neural_dataset_loader"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.train_frac = kwargs.pop("train_frac", 0.8)
        self.dataset = kwargs.pop("dataset", "CSRF19_V1")
        self.sessions_dir = kwargs.pop("sessions_dir", "neuronal_data")
        self.data_dir = "./data/monkey/toliaslab/{}".format(self.dataset)
        self.seed = kwargs.pop("seed", 1000)
        self.subsample = kwargs.pop("subsample", 1)
        self.crop = kwargs.pop("crop", 70)
        self.time_bins_sum = kwargs.pop("time_bins_sum", 12)
        self.target_types = kwargs.pop("target_types", ["neural"])
        self.normalize = kwargs.pop("normalize", True)
        self.stats = kwargs.pop("stats", {})
        self.apply_augmentation = kwargs.pop("apply_augmentation", False)
        self.input_size = kwargs.pop("input_size", 64)
        self.apply_grayscale = kwargs.pop("apply_grayscale", True)
        self.add_fly_corrupted_test = kwargs.pop("add_fly_corrupted_test", {'jpeg_compression': [1, 2, 3, 4, 5], 'pixelate': [1, 2, 3, 4, 5],
                                                  'impulse_noise': [1, 2, 3, 4, 5], 'defocus_blur': [1, 2, 3, 4, 5],
                                                  'contrast': [1, 2, 3, 4, 5], 'fog': [1, 2, 3, 4, 5],
                                                  'brightness': [1, 2, 3, 4, 5], 'frost': [1, 2, 3, 4, 5],
                                                  'glass_blur': [1, 2, 3, 4, 5], 'shot_noise': [1, 2, 3, 4, 5],
                                                  'motion_blur': [1, 2, 3, 4, 5], 'gaussian_noise': [1, 2, 3, 4, 5],
                                                  'elastic_transform': [1, 2, 3, 4, 5], 'snow': [1, 2, 3, 4, 5],
                                                  'zoom_blur': [1, 2, 3, 4, 5]})
        self.resize = kwargs.pop("resize", 0)
        self.update(**kwargs)


class MTLDatasetsConfig(DatasetConfig):
    config_name = "dataset"
    table = Dataset()
    fn = "bias_transfer.dataset.mtl_datasets_loader"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.classification_loader = kwargs.pop("classification_loader", "img_classification")
        self.neural_dataset_dict = kwargs.pop("neural_dataset_dict", {})
        self.neural_dataset_config = NeuralDatasetConfig(
            **self.neural_dataset_dict
        ).to_dict()
        self.img_dataset_dict = kwargs.pop("img_dataset_dict", {})
        if self.classification_loader == "img_classification":
            self.img_dataset_config = ImageDatasetConfig(**self.img_dataset_dict).to_dict()
        else:
            self.img_dataset_config = NeuralDatasetConfig(**self.img_dataset_dict).to_dict()

        self.update(**kwargs)
