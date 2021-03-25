import torchvision
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
from copy import deepcopy

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


def get_module_output(model, input_shape, neural_set):
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
            output = model.to(device)(input, neural_set=neural_set)
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
        v1_model_layer=17, v4_model_layer=-1,
        neural_input_channels=1,
        classification_input_channels=1,
        v1_fine_tune=False,
        momentum=0.1,
        v1_bias=True,
        v1_final_batchnorm=False, add_dropout={},
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
            v1_block = nn.Sequential(
                *list(vgg.features.children())[:v1_model_layer]
            )

        if self.v4_model_layer > 0:
            if self.v1_model_layer > 0:
                v4_block = nn.Sequential(
                    *list(vgg.features.children())[v1_model_layer:self.v4_model_layer]
                )
            else:
                v4_block = nn.Sequential(
                    *list(vgg.features.children())[:v4_model_layer]
                )

        if "shared_block" in self.add_dropout.keys():
            if self.v1_model_layer > 0:
                layers = []
                add_dropout_layer = False
                for child in v1_block:
                    if add_dropout_layer:
                        layers.append(nn.Dropout(self.add_dropout['shared_block']))
                        add_dropout_layer = False
                    layers.append(deepcopy(child))
                    if isinstance(child, nn.modules.activation.ReLU):
                        add_dropout_layer = True
                if isinstance(child, nn.modules.activation.ReLU):
                    layers.append(nn.Dropout(self.add_dropout['shared_block']))

                self.v1_block  = nn.Sequential(*layers)
            if self.v4_model_layer > 0:
                layers = []
                add_dropout_layer = False
                for child in v4_block:
                    if add_dropout_layer:
                        layers.append(nn.Dropout(self.add_dropout['shared_block']))
                        add_dropout_layer = False
                    layers.append(deepcopy(child))
                    if isinstance(child, nn.modules.activation.ReLU):
                        add_dropout_layer = True
                if isinstance(child, nn.modules.activation.ReLU):
                    layers.append(nn.Dropout(self.add_dropout['shared_block']))

                self.v4_block = nn.Sequential(*layers)

        else:
            if self.v1_model_layer > 0:
                self.v1_block = v1_block
            if self.v4_model_layer > 0:
                self.v4_block = v4_block

        # Remove the bias of the last conv layer if not bias:
        if not v1_bias:
            if self.v1_model_layer > 0:
                if "bias" in self.v1_block[-1]._parameters:
                    zeros = torch.zeros_like(self.v1_block[-1].bias)
                    self.v1_block[-1].bias.data = zeros
            if self.v4_model_layer > 0:
                if "bias" in self.v4_block[-1]._parameters:
                    zeros = torch.zeros_like(self.v4_block[-1].bias)
                    self.v4_block[-1].bias.data = zeros

        # Fix pretrained parameters during training parameters
        if not v1_fine_tune:
            if self.v1_model_layer > 0:
                for param in self.v1_block.parameters():
                    param.requires_grad = False
            if self.v4_model_layer > 0:
                for param in self.v4_block.parameters():
                    param.requires_grad = False

        if v1_final_batchnorm:
            if self.v1_model_layer > 0:
                self.v1_extra = nn.Sequential()
                self.v1_extra.add_module(
                    "OutBatchNorm", nn.BatchNorm2d(self.v1_outchannels, momentum=momentum)
                )
                self.v1_extra.add_module("OutNonlin", nn.ReLU(inplace=True))

            if self.v4_model_layer > 0:
                self.v4_extra = nn.Sequential()
                self.v4_extra.add_module(
                    "OutBatchNorm", nn.BatchNorm2d(self.v4_outchannels, momentum=momentum)
                )
                self.v4_extra.add_module("OutNonlin", nn.ReLU(inplace=True))

        if classification:
            if self.v4_model_layer > 0:
                block = nn.Sequential(
                    *list(vgg.features.children())[self.v4_model_layer:]
                )
            else:
                block = nn.Sequential(
                    *list(vgg.features.children())[v1_model_layer:]
                )

            if "unshared_block" in self.add_dropout.keys():
                layers = []
                add_dropout_layer = False
                for child in block:
                    if add_dropout_layer:
                        layers.append(nn.Dropout(self.add_dropout['unshared_block']))
                        add_dropout_layer = False
                    layers.append(deepcopy(child))
                    if isinstance(child, nn.modules.activation.ReLU):
                        add_dropout_layer = True
                if isinstance(child, nn.modules.activation.ReLU):
                    layers.append(nn.Dropout(self.add_dropout['unshared_block']))

                self.unshared_block = nn.Sequential(*layers)
            else:
                self.unshared_block = block

    def forward(self, x, neural_set='v1', classification=False): #, detach_classification_layers=False):
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        if self.v1_model_layer > 0:
            v1_core_out = self.v1_block(x)
            input_next = v1_core_out
            if self.v1_final_batchnorm:
                v1_core_out = self.v1_extra(v1_core_out)
            if neural_set=='v1' and not classification:
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
            if neural_set=='v4' and not classification:
                return v4_core_out, None

        if classification:
            # if detach_classification_layers:
            #     unshared_block_input = shared_core_out.detach()
            # else:
            core_out = self.unshared_block(input_next)
            if self.v4_model_layer > 0:
                return v4_core_out, core_out
            else:
                return v1_core_out, core_out

    @property
    def v1_outchannels(self):
        """
        Returns: dimensions of the output, after a forward pass through the model
        """
        found_out_channels = False
        i = 1
        while not found_out_channels:
            if "out_channels" in self.v1_block[-i].__dict__:
                found_out_channels = True
            else:
                i = i + 1
        return self.v1_block[-i].out_channels

    @property
    def v4_outchannels(self):
        """
        Returns: dimensions of the output, after a forward pass through the model
        """
        found_out_channels = False
        i = 1
        while not found_out_channels:
            if "out_channels" in self.v4_block[-i].__dict__:
                found_out_channels = True
            else:
                i = i + 1
        return self.v4_block[-i].out_channels


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
        v1_model_layer=17, v4_model_layer=-1,
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
        #detach_neural_readout=False,
        add_dropout={}, #detach_classification_layers=False,
        **kwargs
    ):

        super(MTL_VGG, self).__init__()
        self.classification_readout_type = classification_readout_type
        self.input_size = input_size
        self.num_classes = num_classes
        self.v1_elu_offset = v1_elu_offset
        self.neural_input_channels = neural_input_channels
        self.classification_input_channels = classification_input_channels
        #self.detach_neural_readout = detach_neural_readout
        self.add_dropout = add_dropout
        #self.detach_classification_layers = detach_classification_layers

        # for neural dataloaders
        if "v1" in dataloaders["train"].keys():
            v1_train_dataloaders = dataloaders["train"]['v1']
            v1_session_shape_dict = get_dims_for_loader_dict(v1_train_dataloaders)
            names = next(
                iter(list(v1_train_dataloaders.values())[0])
            )._fields
            if  len(names) == 3:
                v1_in_name, _, v1_out_name = names
            else:
                v1_in_name, v1_out_name = names

            self.neural_input_channels = [
                v[v1_in_name][1] for v in v1_session_shape_dict.values()
            ]
            assert (
                np.unique(self.neural_input_channels).size == 1
            ), "all input channels must be of equal size"

        if "v4" in dataloaders["train"].keys():
            v4_train_dataloaders = dataloaders["train"]['v4']
            v4_session_shape_dict = get_dims_for_loader_dict(v4_train_dataloaders)
            names = next(
                iter(list(v4_train_dataloaders.values())[0])
            )._fields
            if  len(names) == 3:
                v4_in_name, _, v4_out_name = names
            else:
                v4_in_name, v4_out_name = names

            self.neural_input_channels = [
                v[v4_in_name][1] for v in v4_session_shape_dict.values()
            ]
            assert (
                np.unique(self.neural_input_channels).size == 1
            ), "all input channels must be of equal size"

        self.mtl_vgg_core = MTL_VGG_Core(
            vgg_type=vgg_type,
            classification=classification,
            pretrained=pretrained,
            v1_model_layer=v1_model_layer, v4_model_layer=v4_model_layer,
            v1_fine_tune=v1_fine_tune,
            neural_input_channels=self.neural_input_channels[0],
            classification_input_channels=self.classification_input_channels,
            v1_final_batchnorm=v1_final_batchnorm,
            v1_bias=v1_bias, add_dropout=self.add_dropout
        )

        if v1_model_layer > 0:
            v1_n_neurons_dict = {k: v[v1_out_name][1] if v1_out_name != "labels" else 1 for k, v in v1_session_shape_dict.items()}
            v1_in_shapes_dict = {k: v[v1_in_name] for k, v in v1_session_shape_dict.items()}
            v1_in_shapes = {}
            for k in v1_n_neurons_dict:
                v1_in_shapes[k] = get_module_output(self.mtl_vgg_core, v1_in_shapes_dict[k], neural_set="v1")[1:]

            self.v1_readout = MultipleGaussian2d(
                in_shapes=v1_in_shapes,
                n_neurons_dict=v1_n_neurons_dict,
                init_mu_range=v1_init_mu_range,
                bias=v1_readout_bias,
                init_sigma_range=v1_init_sigma_range,
                gamma_readout=v1_gamma_readout,
            )
            if v1_readout_bias:
                for key, value in v1_train_dataloaders.items():
                    if v1_out_name == 'labels':
                        targets = getattr(next(iter(value)), v1_out_name).to(torch.float)
                    else:
                        targets = getattr(next(iter(value)), v1_out_name)
                    self.v1_readout[key].bias.data = targets.mean(0)

        if v4_model_layer > 0:
            v4_n_neurons_dict = {k: v[v4_out_name][1] if v4_out_name != "labels" else 1 for k, v in
                                 v4_session_shape_dict.items()}
            v4_in_shapes_dict = {k: v[v4_in_name] for k, v in v4_session_shape_dict.items()}
            v4_in_shapes = {}
            for k in v4_n_neurons_dict:
                v4_in_shapes[k] = get_module_output(self.mtl_vgg_core, v4_in_shapes_dict[k], neural_set="v4")[1:]

            self.v4_readout = MultipleGaussian2d(
                in_shapes=v4_in_shapes,
                n_neurons_dict=v4_n_neurons_dict,
                init_mu_range=v1_init_mu_range,
                bias=v1_readout_bias,
                init_sigma_range=v1_init_sigma_range,
                gamma_readout=v1_gamma_readout,
            )
            if v1_readout_bias:
                for key, value in v4_train_dataloaders.items():
                    if v4_out_name == 'labels':
                        targets = getattr(next(iter(value)), v4_out_name).to(torch.float)
                    else:
                        targets = getattr(next(iter(value)), v4_out_name)
                    self.v4_readout[key].bias.data = targets.mean(0)

        if classification:
            # init fully connected part of vgg
            test_input = Variable(torch.zeros(1, 3, input_size, input_size))
            _, test_out = self.mtl_vgg_core(test_input, neural_set="v4", classification=True)
            self.n_features = test_out.size(1) * test_out.size(2) * test_out.size(3)
            self.classification_readout = create_vgg_readout(
                classification_readout_type,
                n_features=self.n_features,
                num_classes=num_classes,
            )
            self._initialize_weights_classification_readout()

    def forward(self, x, neural_set="v1", data_key=None, classification=False, both=False):
        shared_core_out, core_out = self.mtl_vgg_core(x, neural_set=neural_set, classification=classification) #self.detach_classification_layers)
        if not classification and not both:
            if neural_set=="v1":
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
                # if self.detach_neural_readout:
                #     v1_readout_input = shared_core_out.detach()
                # else:
                if neural_set == "v1":
                    neural_out = self.v1_readout(shared_core_out, data_key=data_key)
                else:
                    neural_out = self.v4_readout(shared_core_out, data_key=data_key)
                neural_out = F.elu(neural_out + self.v1_elu_offset) + 1
                return neural_out, classification_out
            else:
                return classification_out


    def regularizer(self, neural_set="v1", data_key=None):
        if neural_set=="v1":
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
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
