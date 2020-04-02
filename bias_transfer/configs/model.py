from functools import partial

from .base import BaseConfig
from nnfabrik.main import *


class ModelConfig(BaseConfig):
    config_name = "model"
    table = Model()
    fn = "bias_transfer.models.cnn_builder"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cnn_builder = kwargs.pop("cnn_builder", "resnet")
        self.type = kwargs.pop("type", "50")
        self.num_classes = kwargs.pop("num_classes", None)
        self.input_size = kwargs.pop("input_size", 32)
        if not self.num_classes:
            dataset_cls = kwargs.pop("dataset_cls", "CIFAR100")
            if dataset_cls == "CIFAR100":
                self.num_classes = 100
            elif dataset_cls == "CIFAR10":
                self.num_classes = 10
            elif dataset_cls == "TinyImageNet":
                self.num_classes = 200
                self.input_size = 64
            else:
                raise NameError()

        #resnet specific
        self.noise_adv_classification = kwargs.pop("noise_adv_classification", False)
        self.noise_adv_regression = kwargs.pop("noise_adv_regression", False)
        if self.input_size == 32:
            self.core_stride = 1
        elif self.input_size == 64:
            self.core_stride = 2
        self.conv_stem_kernel_size = kwargs.pop("conv_stem_kernel_size", 3)

        # vgg specific
        self.pretrained = kwargs.pop("pretrained", False)
        self.readout_type = kwargs.pop("readout_type", "dense")

        self.update(**kwargs)
