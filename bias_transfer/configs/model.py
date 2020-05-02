from functools import partial

from .base import BaseConfig
from nnfabrik.main import *


class ModelConfig(BaseConfig):
    config_name = "model"
    table = None
    fn = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.update(**kwargs)


class ClassificationModelConfig(ModelConfig):
    config_name = "model"
    table = Model()
    fn = "bias_transfer.models.classification_cnn_builder"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cnn_builder = kwargs.pop("cnn_builder", "resnet")
        self.type = kwargs.pop("type", "50")
        self.num_classes = kwargs.pop("num_classes", None)
        self.input_size = kwargs.pop("input_size", 32)
        self.conv_stem_kernel_size = kwargs.pop("conv_stem_kernel_size", 3)
        self.conv_stem_padding = kwargs.pop("conv_stem_padding", 1)
        self.conv_stem_stride = kwargs.pop("conv_stem_stride", 1)
        self.core_stride = kwargs.pop("core_stride", 1)
        self.max_pool_after_stem = kwargs.pop("max_pool_after_stem", False)
        if not self.num_classes:
            dataset_cls = kwargs.pop("dataset_cls", "CIFAR100")
            if dataset_cls == "CIFAR100":
                self.num_classes = 100
            elif dataset_cls == "CIFAR10":
                self.num_classes = 10
            elif dataset_cls == "TinyImageNet":
                self.num_classes = 200
                self.input_size = 64
                self.core_stride = 2
                self.conv_stem_kernel_size = 5
            elif dataset_cls == "ImageNet":
                self.num_classes = 1000
                self.input_size = 224
                self.conv_stem_kernel_size = 7
                self.conv_stem_padding = 3
                self.conv_stem_stride = 2
                self.max_pool_after_stem = True
            else:
                raise NameError()

        # resnet specific
        self.noise_adv_classification = kwargs.pop("noise_adv_classification", False)
        self.noise_adv_regression = kwargs.pop("noise_adv_regression", False)
        self.get_intermediate_rep = kwargs.pop("get_intermediate_rep", {})
        if (
            self.noise_adv_classification
            or self.noise_adv_regression
            or kwargs.pop("rdm_prediction", False)
            or kwargs.pop("representation_matching", False)
        ):
            self.get_intermediate_rep["flatten"] = "core"

        # vgg specific
        self.pretrained = kwargs.pop("pretrained", False)
        self.readout_type = kwargs.pop("readout_type", "dense")
        self.input_channels = kwargs.pop("input_channels", 3)

        self.update(**kwargs)


class NeuralModelConfig(ModelConfig):
    config_name = "model"
    table = Model()
    fn = "bias_transfer.models.neural_cnn_builder"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.readout_type = kwargs.pop("readout_type", "point")
        if self.readout_type == "point":
            self.hidden_dilation = kwargs.pop("hidden_dilation", 2)
            self.se_reduction = kwargs.pop("se_reduction", 16)
        self.input_kern = kwargs.pop("input_kern", 24)
        self.hidden_kern = kwargs.pop("hidden_kern", 9)
        self.depth_separable = kwargs.pop("depth_separable", True)
        self.stack = kwargs.pop("stack", -1)
        self.n_se_blocks = kwargs.pop("n_se_blocks", 2)
        self.gamma_readout = kwargs.pop("gamma_readout", 0.5)
        self.gamma_input = kwargs.pop("gamma_input", 10)
        self.update(**kwargs)


class MTLModelConfig(ModelConfig):
    config_name = "model"
    table = Model()
    fn = "bias_transfer.models.mtl_builder"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vgg_type = kwargs.pop("vgg_type", "vgg19_bn")
        self.classification = kwargs.pop("classification", False)
        self.classification_readout_type = kwargs.pop(
            "classification_readout_type", None
        )
        self.input_size = kwargs.pop("input_size", None)
        self.num_classes = kwargs.pop("num_classes", 200)
        self.pretrained = kwargs.pop("pretrained", True)

        self.v1_model_layer = kwargs.pop("v1_model_layer", 17)
        self.neural_input_channels = kwargs.pop("neural_input_channels", 1)
        self.v1_fine_tune = kwargs.pop("v1_fine_tune", False)
        self.v1_init_mu_range = kwargs.pop("v1_init_mu_range", 0.3)
        self.v1_init_sigma_range = kwargs.pop("v1_init_sigma_range", 0.6)
        self.v1_readout_bias = kwargs.pop("v1_readout_bias", True)
        self.v1_gamma_readout = kwargs.pop("v1_gamma_readout", 0.5)
        self.v1_elu_offset = kwargs.pop("v1_elu_offset", -1)
        self.classification_input_channels = kwargs.pop(
            "classification_input_channels", 1
        )
        self.update(**kwargs)
