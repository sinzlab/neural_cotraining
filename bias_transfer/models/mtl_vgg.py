from .vgg import VGG_TYPES
import torch.nn as nn
from torch.autograd import Variable
from mlutils.layers.cores import Core2d
import torch
from nnfabrik.utility.nn_helpers import get_dims_for_loader_dict
import numpy as np
from torch.nn import functional as F
from mlutils.layers.legacy import Gaussian2d
from mlutils.training import eval_state
from .vgg import create_vgg_readout


def get_module_output(model, input_shape):
    """
    Gets the output dimensions of the convolutional core
        by passing an input image through all convolutional layers
    :param core: convolutional core of the DNN, which final dimensions
        need to be passed on to the readout layer
    :param input_shape: the dimensions of the input
    :return: output dimensions of the core
    """
    initial_device = "cuda" if next(iter(model.parameters())).is_cuda else "cpu"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with eval_state(model):
        with torch.no_grad():
            input = torch.zeros(1, *input_shape[1:]).to(device)
            output = model.to(device)(input)
    model.to(initial_device)

    return output[0].shape


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
        v1_model_layer=11,
        neural_input_channels=1,
        classification_input_channels=1,
        v1_fine_tune=False,
        momentum=0.1,
        v1_bias=True,
        v1_final_batchnorm=False,
        **kwargs
    ):

        super(MTL_VGG_Core, self).__init__()
        self.v1_model_layer = v1_model_layer
        self.neural_input_channels, self.classification_input_channels = (
            neural_input_channels,
            classification_input_channels,
        )
        self.v1_final_batchnorm = v1_final_batchnorm
        self.classification = classification

        # load convolutional part of vgg
        assert vgg_type in VGG_TYPES, "Unknown vgg_type '{}'".format(vgg_type)
        vgg_loader = VGG_TYPES[vgg_type]
        vgg = vgg_loader(pretrained=pretrained)

        self.shared_block = nn.Sequential(
            *list(vgg.features.children())[:v1_model_layer]
        )

        # Remove the bias of the last conv layer if not bias:
        if not v1_bias:
            if 'bias' in self.shared_block[-1]._parameters:
                zeros = torch.zeros_like(self.shared_block[-1].bias)
                self.shared_block[-1].bias.data = zeros


        # Fix pretrained parameters during training parameters
        if not v1_fine_tune:
            for param in self.shared_block.parameters():
                param.requires_grad = False

        if v1_final_batchnorm:
            self.v1_extra = nn.Sequential()
            self.v1_extra.add_module('OutBatchNorm', nn.BatchNorm2d(self.outchannels, momentum=momentum))
            self.v1_extra.add_module('OutNonlin', nn.ReLU(inplace=True))

        if classification:
            self.unshared_block = nn.Sequential(
                *list(vgg.features.children())[v1_model_layer:]
            )

    def forward(self, x, classification=False):
        if (classification and self.classification_input_channels == 1) or (
            not classification and self.neural_input_channels == 1
        ):
            x = x.expand(-1, 3, -1, -1)
        v1_core_out = shared_core_out = self.shared_block(x)
        if self.v1_final_batchnorm:
            v1_core_out = self.v1_extra(shared_core_out)
        if classification:
            core_out = self.unshared_block(shared_core_out)
            return v1_core_out, core_out
        return v1_core_out, None

    @property
    def outchannels(self):
        """
        Returns: dimensions of the output, after a forward pass through the model
        """
        found_out_channels = False
        i = 1
        while not found_out_channels:
            if 'out_channels' in self.shared_block[-i].__dict__:
                found_out_channels = True
            else:
                i = i + 1
        return self.shared_block[-i].out_channels


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
        v1_model_layer=11,
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
        **kwargs
    ):

        super(MTL_VGG, self).__init__()
        self.classification_readout_type = classification_readout_type
        self.input_size = input_size
        self.num_classes = num_classes
        self.v1_elu_offset = v1_elu_offset
        self.neural_input_channels = neural_input_channels
        self.classification_input_channels = classification_input_channels

        # for neural dataloaders
        if classification:
            neural_train_dataloaders = {
                k: loader
                for k, loader in dataloaders["train"].items()
                if k != "img_classification"
            }
        elif "train" in dataloaders.keys():
            neural_train_dataloaders = dataloaders["train"]
        else:
            neural_train_dataloaders = dataloaders

        session_shape_dict = get_dims_for_loader_dict(neural_train_dataloaders)
        in_name, out_name = next(
            iter(list(neural_train_dataloaders.values())[0])
        )._fields
        self.neural_input_channels = [
            v[in_name][1] for v in session_shape_dict.values()
        ]
        assert (
            np.unique(self.neural_input_channels).size == 1
        ), "all input channels must be of equal size"

        self.mtl_vgg_core = MTL_VGG_Core(
            vgg_type=vgg_type,
            classification=classification,
            pretrained=pretrained,
            v1_model_layer=v1_model_layer,
            v1_fine_tune=v1_fine_tune,
            neural_input_channels=self.neural_input_channels[0],
            classification_input_channels=self.classification_input_channels,
            v1_final_batchnorm=v1_final_batchnorm, v1_bias=v1_bias
        )

        n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
        in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
        in_shapes = {}
        for k in n_neurons_dict:
            in_shapes[k] = get_module_output(self.mtl_vgg_core, in_shapes_dict[k])[1:]

        self.v1_readout = MultipleGaussian2d(
            in_shapes=in_shapes,
            n_neurons_dict=n_neurons_dict,
            init_mu_range=v1_init_mu_range,
            bias=v1_readout_bias,
            init_sigma_range=v1_init_sigma_range,
            gamma_readout=v1_gamma_readout,
        )
        if v1_readout_bias:
            for key, value in neural_train_dataloaders.items():
                _, targets = next(iter(value))
                self.v1_readout[key].bias.data = targets.mean(0)

        if classification:
            # init fully connected part of vgg
            self.classification_readout = create_vgg_readout(
                [self.mtl_vgg_core.shared_block, self.mtl_vgg_core.unshared_block],
                classification_readout_type,
                input_size=input_size,
                num_classes=num_classes,
            )
            self._init_readout_dense()

    def forward(self, x, data_key=None, classification=False):
        shared_core_out, core_out = self.mtl_vgg_core(x, classification)
        if classification:
            if self.classification_readout_type == "dense":
                core_out = core_out.view(core_out.size(0), -1)
            classification_out = self.classification_readout(core_out)
            return classification_out
        v1_out = self.v1_readout(shared_core_out, data_key=data_key)
        v1_out = F.elu(v1_out + self.v1_elu_offset) + 1
        return v1_out

    def regularizer(self, data_key=None):
        return self.v1_readout.regularizer(data_key=data_key)

    def _init_readout_dense(self):
        if self.mtl_vgg_core.classification:
            for m in self.classification_readout:
                if isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()
