from torch import nn
from torchvision.models.resnet import Bottleneck, BasicBlock
import torch
from neuralpredictors.utils import eval_state


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


def freeze_params(model, to_freeze=None, not_to_freeze=None):
    for name, param in model.named_parameters():
        if to_freeze:
            freeze = False
            for freeze_key in to_freeze:
                if freeze_key in name:
                    freeze = True
        elif not_to_freeze:
            freeze = True
            for un_freeze_key in not_to_freeze:
                if un_freeze_key in name:
                    freeze = False
        else:
            raise Exception(
                "Please provide either to_freeze or not_to_freeze arguments!"
            )
        if freeze and param.requires_grad:
            param.requires_grad = False


def freeze_mtl_shared_block(model, multi, tasks):
    if multi:
        if "v1" in tasks:
            for param in model.module.mtl_vgg_core.v1_block.parameters():
                param.requires_grad = False
        if "v4" in tasks:
            for param in model.module.mtl_vgg_core.v4_block.parameters():
                param.requires_grad = False
    else:
        if "v1" in tasks:
            for param in model.mtl_vgg_core.v1_block.parameters():
                param.requires_grad = False
        if "v4" in tasks:
            for param in model.mtl_vgg_core.v4_block.parameters():
                param.requires_grad = False


def weight_reset(m, advanced_init=False, zero_init_residual=False):
    if (
        isinstance(m, nn.Conv1d)
        or isinstance(m, nn.Conv2d)
        or isinstance(m, nn.Linear)
        or isinstance(m, nn.Conv3d)
        or isinstance(m, nn.ConvTranspose1d)
        or isinstance(m, nn.ConvTranspose2d)
        or isinstance(m, nn.ConvTranspose3d)
        or isinstance(m, nn.BatchNorm1d)
        or isinstance(m, nn.BatchNorm2d)
        or isinstance(m, nn.BatchNorm3d)
        or isinstance(m, nn.GroupNorm)
    ):
        m.reset_parameters()
        if advanced_init and isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif advanced_init and isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    if zero_init_residual and isinstance(m, Bottleneck):
        nn.init.constant_(m.bn3.weight, 0)
    elif zero_init_residual and isinstance(m, BasicBlock):
        nn.init.constant_(m.bn2.weight, 0)


def get_model_parameters(model):
    total_parameters = 0
    for layer in list(model.parameters()):
        layer_parameter = 1
        for l in list(layer.size()):
            layer_parameter *= l
        total_parameters += layer_parameter
    return total_parameters


def reset_params(model, reset=None):
    model = model.module if isinstance(model, nn.DataParallel) else model
    if reset == "all":
        print(f"Resetting all parameters")
        model.apply(weight_reset)
    elif reset:
        print(f"Resetting {reset}")
        for name in reset:
            block, layer = name.split(".")[0], int(name.split(".")[1])
            getattr(model, block)[layer].apply(weight_reset)
