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


def cnn_builder(data_loader,
                seed: int,
                **config):
    config = ModelConfig.from_dict(config)
    if config.cnn_builder == "vgg":
        return vgg_builder(seed, config)
    elif config.cnn_builder == "resnet":
        return resnet_builder(seed, config)



def vgg_builder(seed : int, config):
    torch.manual_seed(seed)
    np.random.seed(seed)

    from .vgg import VGG

    model = VGG(input_size=config.input_size, vgg_type=config.type,
                num_classes=config.num_classes, pretrained=config.pretrained,
                classifier_type=config.classifier_type)
    print("Model with {} parameters.".format(get_model_parameters(model)))
    return model


def resnet_builder(seed: int,config):
    #config = ModelConfig.from_dict(config)
    torch.manual_seed(seed)
    np.random.seed(seed)
    type = int(config.type)
    if config.self_attention:
        from .resnet_self_attention import ResNet, Bottleneck
    else:
        from .resnet import ResNet, Bottleneck, BasicBlock
        from .resnet_noise_adv import NoiseAdvResNet

    if type in (18, 34):
        assert not config.self_attention
        block = BasicBlock
    else:
        block = Bottleneck
    if type == 18:
        num_blocks = [2, 2, 2, 2]
    elif type == 26:
        num_blocks = [1, 2, 4, 1]
    elif type == 34:
        num_blocks = [3, 4, 6, 3]
    elif type == 38:
        num_blocks = [2, 3, 5, 2]
    elif type == 50:
        num_blocks = [3, 4, 6, 3]
    elif type == 101:
        num_blocks = [3, 4, 23, 3]
    elif type == 152:
        num_blocks = [3, 8, 36, 3]
    else:
        raise KeyError
    if config.noise_adv_regression or config.noise_adv_classification:
        assert not config.self_attention
        model = NoiseAdvResNet(block, num_blocks, num_classes=config.num_classes,
                              classification=config.noise_adv_classification,
                              adv_readout_layers=config.num_noise_adv_layers,
                               core_stride=config.core_stride,
                               conv_stem_kernel_size=config.conv_stem_kernel_size)
    else:
        model = ResNet(block, num_blocks, num_classes=config.num_classes, core_stride=config.core_stride,
                       conv_stem_kernel_size=config.conv_stem_kernel_size)
    print("Model with {} parameters.".format(get_model_parameters(model)))
    return model
