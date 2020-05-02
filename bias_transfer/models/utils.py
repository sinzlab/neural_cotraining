from torch import nn


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
        if freeze:
            param.requires_grad = False


def weight_reset(m):
    if (
        isinstance(m, nn.Conv2d)
        or isinstance(m, nn.Linear)
        or isinstance(m, nn.Conv3d)
        or isinstance(m, nn.ConvTranspose1d)
        or isinstance(m, nn.ConvTranspose2d)
        or isinstance(m, nn.ConvTranspose3d)
        or isinstance(m, nn.BatchNorm1d)
        or isinstance(m, nn.BatchNorm2d)
        or isinstance(m, nn.BatchNorm3d)
    ):
        m.reset_parameters()


def get_model_parameters(model):
    total_parameters = 0
    for layer in list(model.parameters()):
        layer_parameter = 1
        for l in list(layer.size()):
            layer_parameter *= l
        total_parameters += layer_parameter
    return total_parameters