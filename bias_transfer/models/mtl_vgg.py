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

def get_module_output(model, input_shape):
    """
    Gets the output dimensions of the convolutional core
        by passing an input image through all convolutional layers
    :param core: convolutional core of the DNN, which final dimensions
        need to be passed on to the readout layer
    :param input_shape: the dimensions of the input
    :return: output dimensions of the core
    """
    initial_device = 'cuda' if next(iter(model.parameters())).is_cuda else 'cpu'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with eval_state(model):
        with torch.no_grad():
            input = torch.zeros(1, *input_shape[1:]).to(device)
            output = model.to(device)(input)
    model.to(initial_device)

    return output[0].shape

class MultipleGaussian2d(torch.nn.ModuleDict):
    def __init__(self, in_shapes, n_neurons_dict, init_mu_range, init_sigma_range, bias, gamma_readout):
        # super init to get the _module attribute
        super(MultipleGaussian2d, self).__init__()
        for k in n_neurons_dict:
            in_shape = in_shapes[k]
            n_neurons = n_neurons_dict[k]
            self.add_module(k, Gaussian2d(
                in_shape=in_shape,
                outdims=n_neurons,
                init_mu_range=init_mu_range,
                init_sigma_range=init_sigma_range,
                bias=bias)
                            )
        self.gamma_readout = gamma_readout

    def forward(self, *args, data_key=None, **kwargs):
        if data_key is None and len(self) == 1:
            data_key = list(self.keys())[0]
        return self[data_key](*args, **kwargs)

    def regularizer(self, data_key):
        return self[data_key].feature_l1(average=False) * self.gamma_readout




class MTL_VGG_Core(Core2d, nn.Module):

    def __init__(self, classification=True, vgg_type='vgg19_bn', pretrained=True,
                 v1_model_layer=17, v1_final_batchnorm=True, input_channels=1,
                 v1_final_nonlinearity=True, v1_bias=False, v1_momentum=0.1, v1_fine_tune = False,
                 **kwargs):

        super(MTL_VGG_Core, self).__init__()
        self.classification = classification
        self.v1_model_layer = v1_model_layer
        self.input_channels = input_channels


        # load convolutional part of vgg
        assert vgg_type in VGG_TYPES, "Unknown vgg_type '{}'".format(vgg_type)
        vgg_loader = VGG_TYPES[vgg_type]
        vgg = vgg_loader(pretrained=pretrained)

        self.shared_block = nn.Sequential(*list(vgg.features.children())[:v1_model_layer])
        print(self.shared_block)
        # Fix pretrained parameters during training parameters
        if not v1_fine_tune:
            for param in self.shared_block.parameters():
                param.requires_grad = False

        # self.v1_core.add_module('V1_CORE', self.v1_vgg_layers)
        # if v1_final_batchnorm:
        #     self.v1_core.add_module('OutBatchNorm', nn.BatchNorm2d(self.outchannels, momentum=v1_momentum))
        # if v1_final_nonlinearity:
        #     self.v1_core.add_module('OutNonlin', nn.ReLU(inplace=True))


        if classification:
            self.unshared_block = nn.Sequential(*list(vgg.features.children())[v1_model_layer:])


    def forward(self, x, **kwargs):
        if self.input_channels == 1:
            x = x.expand(-1, 3, -1, -1)
            #core_out = self.v1_core(x)
        shared_core_out = self.shared_block(x)
        if self.classification:
            core_out = self.unshared_block(shared_core_out)
            return shared_core_out, core_out
        return shared_core_out, None

    #
    # @property
    # def outchannels(self):
    #     """
    #     Returns: dimensions of the output, after a forward pass through the model
    #     """
    #     found_out_channels = False
    #     i=1
    #     while not found_out_channels:
    #         if 'out_channels' in self.v1_core.V1_CORE[-i].__dict__:
    #             found_out_channels = True
    #         else:
    #             i = i+1
    #     return self.v1_core.V1_CORE[-i].out_channels


class MTL_VGG(nn.Module):

    def __init__(self, dataloaders, vgg_type='vgg19_bn', classification=False,
                 classification_readout_type=None, input_size=None, num_classes=200, pretrained=True,
                 v1_model_layer=17, v1_final_batchnorm=True, input_channels=1,
                 v1_final_nonlinearity=True, v1_bias=False, v1_momentum=0.1, v1_fine_tune = False,
                 v1_init_mu_range=0.4, v1_init_sigma_range=0.6, v1_readout_bias=True,
                 v1_gamma_readout=0.002, v1_elu_offset=-1, **kwargs):

        super(MTL_VGG, self).__init__()
        self.classification_readout_type = classification_readout_type
        self.input_size = input_size
        self.num_classes = num_classes
        self.v1_elu_offset = v1_elu_offset
        self.input_channels = input_channels

        #for neural dataloaders
        if "train" in dataloaders.keys():
            dataloaders = dataloaders["train"]
        session_shape_dict = get_dims_for_loader_dict(dataloaders)
        in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields
        self.input_channels = [v[in_name][1] for v in session_shape_dict.values()]
        assert np.unique(self.input_channels).size == 1, "all input channels must be of equal size"

        self.mtl_vgg_core = MTL_VGG_Core(vgg_type=vgg_type, classification=classification, pretrained=pretrained,
                                         v1_model_layer=v1_model_layer, v1_final_batchnorm=v1_final_batchnorm,
                                         v1_final_nonlinearity=v1_final_nonlinearity, v1_bias=v1_bias,
                                         v1_momentum=v1_momentum,
                                         v1_fine_tune=v1_fine_tune, input_channels=self.input_channels[0])

        n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
        in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
        in_shapes = {}
        for k in n_neurons_dict:
            in_shapes[k] = get_module_output(self.mtl_vgg_core, in_shapes_dict[k])[1:]

        self.v1_readout = MultipleGaussian2d(in_shapes=in_shapes,
                             n_neurons_dict=n_neurons_dict,
                             init_mu_range=v1_init_mu_range,
                             bias=v1_readout_bias,
                             init_sigma_range=v1_init_sigma_range,
                             gamma_readout=v1_gamma_readout)
        if v1_readout_bias:
            for key, value in dataloaders.items():
                _, targets = next(iter(value))
                self.v1_readout[key].bias.data = targets.mean(0)


        if self.mtl_vgg_core.classification:
            # init fully connected part of vgg
            if classification_readout_type == "dense":
                test_input = Variable(torch.zeros(1, 3, input_size, input_size))
                test_out = self.mtl_vgg_core.vgg_core.features(test_input)
                self.classification_core_out_flat = test_out.size(1) * test_out.size(2) * test_out.size(3)
                self.classification_readout = nn.Sequential(nn.Linear(self.classification_core_out_flat, 4096),
                                             nn.ReLU(True),
                                             nn.Dropout(),
                                             nn.Linear(4096, 4096),
                                             nn.ReLU(True),
                                             nn.Dropout(),
                                             nn.Linear(4096, num_classes)
                                             )
                self._init_readout_dense()
            elif classification_readout_type == "conv":
                self.classification_readout = nn.Sequential(nn.Conv2d(512, 4096, 1, 1, bias=True),
                                             nn.ReLU(True),
                                             nn.Dropout(),
                                             nn.Conv2d(4096, 4096, 1, 1, bias=True),
                                             nn.ReLU(True),
                                             nn.Dropout(),
                                             nn.Conv2d(4096, num_classes, 1, 1, bias=True),
                                             nn.AdaptiveMaxPool2d(1), nn.Flatten())






    def forward(self, x, data_key=None, **kwargs):
        shared_core_out, core_out = self.mtl_vgg_core(x)
        v1_out = self.v1_readout(shared_core_out, data_key=data_key)
        v1_out = F.elu(v1_out + self.v1_elu_offset) + 1
        if core_out:
            if self.classification_readout_type == "dense":
                core_out = core_out.view(core_out.size(0), -1)
            class_out = self.classification_readout(core_out)
            classification_out = {"logits": class_out}
            return v1_out, classification_out
        return v1_out


    def regularizer(self, data_key=None):
        return self.v1_readout.regularizer(data_key=data_key)

    def _init_readout_dense(self):
        if self.mtl_vgg_core.classification:
            for m in self.classification_readout:
                if isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()
