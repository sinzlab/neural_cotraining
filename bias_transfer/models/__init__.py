import torch
import numpy as np

from bias_transfer.configs.model import ClassificationModelConfig, MTLModelConfig
from nnvision.models.models import se_core_gauss_readout, se_core_point_readout


def get_model_parameters(model):
    total_parameters = 0
    for layer in list(model.parameters()):
        layer_parameter = 1
        for l in list(layer.size()):
            layer_parameter *= l
        total_parameters += layer_parameter
    return total_parameters


def neural_cnn_builder(data_loaders, seed: int = 1000, **config):
    config.pop("comment", None)
    readout_type = config.pop("readout_type", None)
    if readout_type == "point":
        model = se_core_point_readout(dataloaders=data_loaders, seed=seed, **config)
    elif readout_type == "gauss":
        model = se_core_gauss_readout(dataloaders=data_loaders, seed=seed, **config)
    print("Model with {} parameters.".format(get_model_parameters(model)))
    return model


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
        neural_input_channels=config.neural_input_channels,
        classification_input_channels=config.classification_input_channels,
        v1_fine_tune=config.v1_fine_tune,
        v1_init_mu_range=config.v1_init_mu_range,
        v1_init_sigma_range=config.v1_init_sigma_range,
        v1_readout_bias=config.v1_readout_bias,
        v1_gamma_readout=config.v1_gamma_readout,
        v1_elu_offset=config.v1_elu_offset,
    )

    print("Model with {} parameters.".format(get_model_parameters(model)))
    return model


def classification_cnn_builder(data_loader, seed: int, **config):
    config = ClassificationModelConfig.from_dict(config)
    if config.cnn_builder == "vgg":
        return vgg_builder(seed, config)
    elif config.cnn_builder == "resnet":
        return resnet_builder(seed, config)


def vgg_builder(seed: int, config):
    torch.manual_seed(seed)
    np.random.seed(seed)

    from .vgg import VGG

    model = VGG(
        input_size=config.input_size,
        vgg_type=config.type,
        num_classes=config.num_classes,
        pretrained=config.pretrained,
        readout_type=config.readout_type,
        input_channels=config.input_channels,
    )
    print("Model with {} parameters.".format(get_model_parameters(model)))
    return model


def resnet_builder(seed: int, config):
    # config = ModelConfig.from_dict(config)
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
        model = NoiseAdvResNet(
            block,
            num_blocks,
            num_classes=config.num_classes,
            classification=config.noise_adv_classification,
            adv_readout_layers=config.num_noise_adv_layers,
            core_stride=config.core_stride,
            conv_stem_kernel_size=config.conv_stem_kernel_size,
        )
    else:
        model = ResNet(
            block,
            num_blocks,
            num_classes=config.num_classes,
            core_stride=config.core_stride,
            conv_stem_kernel_size=config.conv_stem_kernel_size,
        )
    print("Model with {} parameters.".format(get_model_parameters(model)))
    return model
