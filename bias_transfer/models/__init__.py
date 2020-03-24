import torch
import numpy as np

from bias_transfer.configs.model import ModelConfig


def get_model_parameters(model):
    total_parameters = 0
    for layer in list(model.parameters()):
        layer_parameter = 1
        for l in list(layer.size()):
            layer_parameter *= l
        total_parameters += layer_parameter
    return total_parameters


def resnet_builder(data_loader,
                   seed: int,
                   **config):
    config = ModelConfig.from_dict(config)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if config.self_attention:
        from .resnet_self_attention import ResNet, Bottleneck
    else:
        from .resnet import ResNet, Bottleneck, BasicBlock
        from .resnet_noise_adv import NoiseAdvResNet

    if config.type in (18, 34):
        assert not config.self_attention
        block = BasicBlock
    else:
        block = Bottleneck
    if config.type == 18:
        num_blocks = [2, 2, 2, 2]
    elif config.type == 26:
        num_blocks = [1, 2, 4, 1]
    elif config.type == 34:
        num_blocks = [3, 4, 6, 3]
    elif config.type == 38:
        num_blocks = [2, 3, 5, 2]
    elif config.type == 50:
        num_blocks = [3, 4, 6, 3]
    elif config.type == 101:
        num_blocks = [3, 4, 23, 3]
    elif config.type == 152:
        num_blocks = [3, 8, 36, 3]
    else:
        raise KeyError
    if config.noise_adv_regression or config.noise_adv_classification:
        assert not config.self_attention
        model = NoiseAdvResNet(block, num_blocks, num_classes=config.num_classes,
                              classification=config.noise_adv_classification,
                              adv_readout_layers=config.num_noise_adv_layers)
    else:
        model = ResNet(block, num_blocks, num_classes=config.num_classes)
    print("Model with {} parameters.".format(get_model_parameters(model)))
    return model
