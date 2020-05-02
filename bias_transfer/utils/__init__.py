from torch import nn


def stringify(x):
    if type(x) is dict:
        x = ".".join(["{}_{}".format(k, v) for k, v in x.items()])
    return str(x)


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
