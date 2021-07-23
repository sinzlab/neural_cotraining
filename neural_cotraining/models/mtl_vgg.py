import torchvision
import torch.nn as nn
from torch.autograd import Variable
from neuralpredictors.layers.cores.base import Core2d
import torch
from nnfabrik.utility.nn_helpers import get_dims_for_loader_dict
import numpy as np
from torch.nn import functional as F
from mlutils.layers.legacy import Gaussian2d
from .vgg import create_vgg_readout
from copy import deepcopy
from neural_cotraining.models.utils import get_model_parameters
from neural_cotraining.configs.model import MTLModelConfig
from .utils import get_module_output

VGG_TYPES = {
    "vgg11": torchvision.models.vgg11,
    "vgg11_bn": torchvision.models.vgg11_bn,
    "vgg13": torchvision.models.vgg13,
    "vgg13_bn": torchvision.models.vgg13_bn,
    "vgg16": torchvision.models.vgg16,
    "vgg16_bn": torchvision.models.vgg16_bn,
    "vgg19_bn": torchvision.models.vgg19_bn,
    "vgg19": torchvision.models.vgg19,
}


class MultipleGaussian2d(torch.nn.ModuleDict):
    def __init__(
        self,
        in_shapes,
        n_neurons_dict,
        init_mu_range,
        init_sigma_range,
        bias,
        gamma_readout,
    ):
        # super init to get the _module attribute
        super(MultipleGaussian2d, self).__init__()
        for k in n_neurons_dict:
            in_shape = in_shapes[k]
            n_neurons = n_neurons_dict[k]
            self.add_module(
                k,
                Gaussian2d(
                    in_shape=in_shape,
                    outdims=n_neurons,
                    init_mu_range=init_mu_range,
                    init_sigma_range=init_sigma_range,
                    bias=bias,
                ),
            )
        self.gamma_readout = gamma_readout

    def forward(self, *args, data_key=None, **kwargs):
        if data_key is None and len(self) == 1:
            data_key = list(self.keys())[0]
        return self[data_key](*args, **kwargs)

    def regularizer(self, data_key):
        return self[data_key].feature_l1(average=False) * self.gamma_readout


class MTL_VGG_Core(Core2d, nn.Module):
    def __init__(
        self,
        classification=True,
        vgg_type="vgg19_bn",
        pretrained=True,
        v1_model_layer=17,
        v4_model_layer=-1,
        neural_input_channels=1,
        classification_input_channels=1,
        v1_fine_tune=False,
        momentum=0.1,
        v1_bias=True,
        v1_final_batchnorm=False,
        add_dropout={},
        **kwargs
    ):

        super(MTL_VGG_Core, self).__init__()
        self.v1_model_layer = v1_model_layer
        self.v4_model_layer = v4_model_layer
        self.neural_input_channels, self.classification_input_channels = (
            neural_input_channels,
            classification_input_channels,
        )
        self.v1_final_batchnorm = v1_final_batchnorm
        self.classification = classification
        self.add_dropout = add_dropout

        # load convolutional part of vgg
        assert vgg_type in VGG_TYPES, "Unknown vgg_type '{}'".format(vgg_type)
        vgg_loader = VGG_TYPES[vgg_type]
        vgg = vgg_loader(pretrained=pretrained)

        if self.v1_model_layer > 0:
            v1_block = nn.Sequential(*list(vgg.features.children())[:v1_model_layer])

        if self.v4_model_layer > 0:
            if self.v1_model_layer > 0:
                v4_block = nn.Sequential(
                    *list(vgg.features.children())[v1_model_layer : self.v4_model_layer]
                )
            else:
                v4_block = nn.Sequential(
                    *list(vgg.features.children())[:v4_model_layer]
                )

        if "shared_block" in self.add_dropout.keys():
            if self.v1_model_layer > 0:
                self.v1_block = self.add_dropout_layers(
                    v1_block, self.add_dropout["shared_block"]
                )
            if self.v4_model_layer > 0:
                self.v4_block = self.add_dropout_layers(
                    v4_block, self.add_dropout["shared_block"]
                )

        else:
            if self.v1_model_layer > 0:
                self.v1_block = v1_block
            if self.v4_model_layer > 0:
                self.v4_block = v4_block

        # Remove the bias of the last conv layer if not bias:
        if not v1_bias:
            if self.v1_model_layer > 0:
                self.remove_bias(self.v1_block)
            if self.v4_model_layer > 0:
                self.remove_bias(self.v4_block)

        # Fix pretrained parameters during training parameters
        if not v1_fine_tune:
            if self.v1_model_layer > 0:
                self.fix_weights(self.v1_block)
            if self.v4_model_layer > 0:
                self.fix_weights(self.v4_block)

        if v1_final_batchnorm:
            if self.v1_model_layer > 0:
                self.v1_extra = self.add_bn(self.outchannels(self.v1_block), momentum)

            if self.v4_model_layer > 0:
                self.v4_extra = self.add_bn(self.outchannels(self.v4_block), momentum)

        if classification:
            if self.v4_model_layer > 0:
                block = nn.Sequential(
                    *list(vgg.features.children())[self.v4_model_layer :]
                )
            else:
                block = nn.Sequential(*list(vgg.features.children())[v1_model_layer:])

            if "unshared_block" in self.add_dropout.keys():
                self.unshared_block = self.add_dropout_layers(
                    block, self.add_dropout["unshared_block"]
                )
            else:
                self.unshared_block = block

    def forward(self, x, neural_set="v1", classification=False):
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        if self.v1_model_layer > 0:
            v1_core_out = self.v1_block(x)
            input_next = v1_core_out
            if self.v1_final_batchnorm:
                v1_core_out = self.v1_extra(v1_core_out)
            if neural_set == "v1" and not classification:
                return v1_core_out, None

        if self.v4_model_layer > 0:
            if self.v1_model_layer > 0:
                v4_core_out = self.v4_block(input_next)
                input_next = v4_core_out
                if self.v1_final_batchnorm:
                    v4_core_out = self.v4_extra(v4_core_out)
            else:
                v4_core_out = self.v4_block(x)
                input_next = v4_core_out
                if self.v1_final_batchnorm:
                    v4_core_out = self.v4_extra(v4_core_out)
            if neural_set == "v4" and not classification:
                return v4_core_out, None

        if classification:
            core_out = self.unshared_block(input_next)
            if self.v4_model_layer > 0:
                return v4_core_out, core_out
            else:
                return v1_core_out, core_out

    @property
    def outchannels(self, block):
        """
        Returns: dimensions of the output, after a forward pass through the model
        """
        found_out_channels = False
        i = 1
        while not found_out_channels:
            if "out_channels" in block[-i].__dict__:
                found_out_channels = True
            else:
                i = i + 1
        return block[-i].out_channels

    def add_dropout_layers(self, block, dropout_rate):
        layers = []
        add_dropout_layer = False
        for child in block:
            if add_dropout_layer:
                layers.append(nn.Dropout(dropout_rate))
                add_dropout_layer = False
            layers.append(deepcopy(child))
            if isinstance(child, nn.modules.activation.ReLU):
                add_dropout_layer = True
        if isinstance(child, nn.modules.activation.ReLU):
            layers.append(nn.Dropout(dropout_rate))

        return nn.Sequential(*layers)

    def remove_bias(self, block):
        if "bias" in block[-1]._parameters:
            zeros = torch.zeros_like(block[-1].bias)
            block[-1].bias.data = zeros

    def fix_weights(self, block):
        for param in block.parameters():
            param.requires_grad = False

    def add_bn(self, outchannels, momentum):
        extra = nn.Sequential()
        extra.add_module("OutBatchNorm", nn.BatchNorm2d(outchannels, momentum=momentum))
        extra.add_module("OutNonlin", nn.ReLU(inplace=True))
        return extra


class MTL_VGG(nn.Module):
    def __init__(
        self,
        dataloaders,
        vgg_type="vgg19_bn",
        classification=False,
        classification_readout_type=None,
        input_size=None,
        num_classes=200,
        pretrained=True,
        v1_model_layer=17,
        v4_model_layer=-1,
        neural_input_channels=1,
        classification_input_channels=1,
        v1_fine_tune=False,
        v1_init_mu_range=0.4,
        v1_init_sigma_range=0.6,
        v1_readout_bias=True,
        v1_bias=True,
        v1_final_batchnorm=False,
        v1_gamma_readout=0.002,
        v1_elu_offset=-1,
        add_dropout={},
        **kwargs
    ):

        super(MTL_VGG, self).__init__()
        self.classification_readout_type = classification_readout_type
        self.input_size = input_size
        self.num_classes = num_classes
        self.v1_elu_offset = v1_elu_offset
        self.neural_input_channels = neural_input_channels
        self.classification_input_channels = classification_input_channels
        self.add_dropout = add_dropout
        self.dataloaders = dataloaders
        self.v1_init_sigma_range = v1_init_sigma_range
        self.v1_init_mu_range = v1_init_mu_range
        self.v1_readout_bias = v1_readout_bias
        self.v1_gamma_readout = v1_gamma_readout

        # for neural dataloaders
        if "v1" in dataloaders["train"].keys():
            self.set_neural_loader_info("v1")

        if "v4" in dataloaders["train"].keys():
            self.set_neural_loader_info("v4")

        self.mtl_vgg_core = MTL_VGG_Core(
            vgg_type=vgg_type,
            classification=classification,
            pretrained=pretrained,
            v1_model_layer=v1_model_layer,
            v4_model_layer=v4_model_layer,
            v1_fine_tune=v1_fine_tune,
            neural_input_channels=self.neural_input_channels[0],
            classification_input_channels=self.classification_input_channels,
            v1_final_batchnorm=v1_final_batchnorm,
            v1_bias=v1_bias,
            add_dropout=self.add_dropout,
        )

        if v1_model_layer > 0:
            self.create_gaussian_readout("v1")

        if v4_model_layer > 0:
            self.create_gaussian_readout("v4")

        if classification:
            # init fully connected part of vgg
            test_input = Variable(torch.zeros(1, 3, input_size, input_size))
            _, test_out = self.mtl_vgg_core(
                test_input, neural_set="v4", classification=True
            )
            self.n_features = test_out.size(1) * test_out.size(2) * test_out.size(3)
            self.classification_readout = create_vgg_readout(
                classification_readout_type,
                n_features=self.n_features,
                num_classes=num_classes,
            )
            self._initialize_weights_classification_readout()

    def forward(
        self, x, neural_set="v1", data_key=None, classification=False, both=False
    ):
        shared_core_out, core_out = self.mtl_vgg_core(
            x, neural_set=neural_set, classification=classification
        )  # self.detach_classification_layers)
        if not classification and not both:
            if neural_set == "v1":
                neural_out = self.v1_readout(shared_core_out, data_key=data_key)
            else:
                neural_out = self.v4_readout(shared_core_out, data_key=data_key)
            neural_out = F.elu(neural_out + self.v1_elu_offset) + 1
            return neural_out
        else:
            if self.classification_readout_type == "dense":
                core_out = core_out.view(core_out.size(0), -1)
            classification_out = self.classification_readout(core_out)
            if both:
                if neural_set == "v1":
                    neural_out = self.v1_readout(shared_core_out, data_key=data_key)
                else:
                    neural_out = self.v4_readout(shared_core_out, data_key=data_key)
                neural_out = F.elu(neural_out + self.v1_elu_offset) + 1
                return neural_out, classification_out
            else:
                return classification_out

    def regularizer(self, neural_set="v1", data_key=None):
        if neural_set == "v1":
            return self.v1_readout.regularizer(data_key=data_key)
        else:
            return self.v4_readout.regularizer(data_key=data_key)

    def freeze(self, selection=("v1",)):
        if "v1" in selection:
            for param in self.mtl_vgg_core.v1_block.parameters():
                param.requires_grad = False
        if "v4" in selection:
            for param in self.mtl_vgg_core.v4_block.parameters():
                param.requires_grad = False

    def _initialize_weights_classification_readout(self):
        if self.mtl_vgg_core.classification:
            for m in self.classification_readout:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def set_neural_loader_info(self, neural_set):
        setattr(
            self,
            "{}_train_dataloaders".format(neural_set),
            self.dataloaders["train"][neural_set],
        )
        session_shape_dict = get_dims_for_loader_dict(
            getattr(self, "{}_train_dataloaders".format(neural_set))
        )
        setattr(self, "{}_session_shape_dict".format(neural_set), session_shape_dict)
        names = next(
            iter(
                list(getattr(self, "{}_train_dataloaders".format(neural_set)).values())[
                    0
                ]
            )
        )._fields
        if len(names) == 3:
            in_name, _, out_name = names
        else:
            in_name, out_name = names
        setattr(self, "{}_in_name".format(neural_set), in_name)
        setattr(self, "{}_out_name".format(neural_set), out_name)
        self.neural_input_channels = [
            v[in_name][1] for v in session_shape_dict.values()
        ]
        assert (
            np.unique(self.neural_input_channels).size == 1
        ), "all input channels must be of equal size"

    def create_gaussian_readout(self, neural_set):
        in_name = getattr(self, "{}_in_name".format(neural_set))
        out_name = getattr(self, "{}_out_name".format(neural_set))
        session_shape_dict = getattr(self, "{}_session_shape_dict".format(neural_set))
        n_neurons_dict = {
            k: v[out_name][1] if out_name != "labels" else 1
            for k, v in session_shape_dict.items()
        }
        in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
        in_shapes = {}
        for k in n_neurons_dict:
            in_shapes[k] = get_module_output(
                self.mtl_vgg_core, in_shapes_dict[k], neural_set=neural_set
            )[1:]

        readout = MultipleGaussian2d(
            in_shapes=in_shapes,
            n_neurons_dict=n_neurons_dict,
            init_mu_range=self.v1_init_mu_range,
            bias=self.v1_readout_bias,
            init_sigma_range=self.v1_init_sigma_range,
            gamma_readout=self.v1_gamma_readout,
        )
        if self.v1_readout_bias:
            train_dataloaders = getattr(self, "{}_train_dataloaders".format(neural_set))
            for key, value in train_dataloaders.items():
                if out_name == "labels":
                    targets = getattr(next(iter(value)), out_name).to(torch.float)
                else:
                    targets = getattr(next(iter(value)), out_name)
                readout[key].bias.data = targets.mean(0)
        setattr(self, "{}_readout".format(neural_set), readout)


def mtl_builder(data_loaders, seed: int = 1000, **config):
    config = MTLModelConfig.from_dict(config)
    torch.manual_seed(seed)
    np.random.seed(seed)

    from .mtl_vgg import MTL_VGG

    model = MTL_VGG(
        data_loaders,
        vgg_type=config.vgg_type,
        classification=config.classification,
        classification_readout_type=config.classification_readout_type,
        input_size=config.input_size,
        num_classes=config.num_classes,
        pretrained=config.pretrained,
        v1_model_layer=config.v1_model_layer,
        v4_model_layer=config.v4_model_layer,
        neural_input_channels=config.neural_input_channels,
        classification_input_channels=config.classification_input_channels,
        v1_fine_tune=config.v1_fine_tune,
        v1_init_mu_range=config.v1_init_mu_range,
        v1_init_sigma_range=config.v1_init_sigma_range,
        v1_readout_bias=config.v1_readout_bias,
        v1_bias=config.v1_bias,
        v1_gamma_readout=config.v1_gamma_readout,
        v1_elu_offset=config.v1_elu_offset,
        v1_final_batchnorm=config.v1_final_batchnorm,
        add_dropout=config.add_dropout,
    )

    print("Model with {} parameters.".format(get_model_parameters(model)))
    return model
