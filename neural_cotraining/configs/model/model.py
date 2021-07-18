from functools import partial

from nntransfer.configs.model.base import ModelConfig
from nntransfer.tables.nnfabrik import Model


class ClassificationModelConfig(ModelConfig):
    config_name = "model"
    table = Model()
    fn = "neural_cotraining.models.classification_cnn_builder"

    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)

        self.type : str = "vgg19_bn"
        self.input_size : int = 64
        self.avg_pool : bool = False
        self.dataset_cls : str = "CIFAR100"
        if self.dataset_cls == "CIFAR100":
            self.num_classes : int = 100
        elif self.dataset_cls == "CIFAR10":
            self.num_classes : int = 10
        elif self.dataset_cls == "TinyImageNet":
            self.num_classes : int = 200
            self.input_size = 64
        elif self.dataset_cls == "ImageNet":
            self.num_classes : int = 1000
            self.input_size = 224
            self.avg_pool = True
        elif self.dataset_cls == "V1_ImageNet":
            self.num_classes : int = 964
            self.input_size = 93
        else:
            raise NameError()

        # vgg specific
        self.pretrained : bool = False
        self.readout_type : str = "dense"
        self.input_channels : int = 3

        super().__init__(**kwargs)


class NeuralModelConfig(ModelConfig):
    config_name = "model"
    table = Model()
    fn = "neural_cotraining.models.neural_cnn_builder"

    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.readout_type : str = "point"
        if self.readout_type == "point":
            self.hidden_dilation : int = 2
            self.se_reduction : int = 16
        self.input_kern : int = 24
        self.hidden_kern : int = 9
        self.depth_separable : bool = True
        self.stack : int = -1
        self.n_se_blocks : int = 2
        self.gamma_readout : float = 0.5
        self.gamma_input : int = 10
        super().__init__(**kwargs)


class MTLModelConfig(ModelConfig):
    config_name = "model"
    table = Model()
    fn = "neural_cotraining.models.mtl_builder"

    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)

        self.vgg_type : str = "vgg19_bn"
        self.classification : bool = False
        self.classification_readout_type : str = None

        self.input_size : int = None
        self.num_classes : int = 200
        self.pretrained : bool = True
        self.add_dropout : dict = {}
        #self.detach_classification_layers = kwargs.pop("detach_classification_layers", False)

        #self.detach_neural_readout = kwargs.pop("detach_neural_readout", False)
        self.v1_model_layer : int = 17
        self.v4_model_layer : int = -1
        self.neural_input_channels : int = 1
        self.v1_fine_tune : bool = False
        self.v1_init_mu_range : float = 0.3
        self.v1_init_sigma_range : float = 0.6
        self.v1_readout_bias : bool = True
        self.v1_bias : bool = True
        self.v1_final_batchnorm : bool = False
        self.v1_gamma_readout : float = 0.5
        self.v1_elu_offset : int = -1
        self.classification_input_channels : int = 1
        super().__init__(**kwargs)
