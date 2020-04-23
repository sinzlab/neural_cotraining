from torch import nn


def stringify(x):
    if type(x) is dict:
        x = ".".join(["{}_{}".format(k, v) for k, v in x.items()])
    return str(x)


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()
