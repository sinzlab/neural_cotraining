from functools import partial

from .base import BaseConfig
from nnfabrik.main import *


class ModelConfig(BaseConfig):
    config_name = "model"
    table = Model()
    fn = "bias_transfer.models.resnet_builder"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.noise_adv_classification = kwargs.pop("noise_adv_classification", False)
        self.noise_adv_regression = kwargs.pop("noise_adv_regression", False)
        self.type = kwargs.pop("type", 50)
        self.num_classes = kwargs.pop("num_classes", 100)
        self.update(**kwargs)


class CIFAR100(ModelConfig):
    pass


class CIFAR10(ModelConfig):
    def __init__(self, **kwargs):
        kwargs.pop("num_classes", None)
        super().__init__(num_classes=10, **kwargs)

