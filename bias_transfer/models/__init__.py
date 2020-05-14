import torch
import numpy as np

from bias_transfer.configs.model import ClassificationModelConfig, MTLModelConfig
from bias_transfer.models.resnet import resnet_builder
from bias_transfer.models.wrappers.noise_adv import NoiseAdvWrapper
from bias_transfer.models.utils import get_model_parameters
from bias_transfer.models.vgg import vgg_builder
from torch.hub import load_state_dict_from_url
from nnvision.models.models import se_core_gauss_readout, se_core_point_readout
from .wrappers import *


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
        v1_bias=config.v1_bias,
        v1_gamma_readout=config.v1_gamma_readout,
        v1_elu_offset=config.v1_elu_offset,
        v1_final_batchnorm=config.v1_final_batchnorm,
    )

    print("Model with {} parameters.".format(get_model_parameters(model)))
    return model


def classification_cnn_builder(data_loader, seed: int, **config):
    config = ClassificationModelConfig.from_dict(config)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if "vgg" in config.type:
        model = vgg_builder(seed, config)
        from torchvision.models.vgg import model_urls
    elif "resnet" in config.type:
        model = resnet_builder(seed, config)
        from torchvision.models.resnet import model_urls
    else:
        raise Exception("Unknown type {}".format(config.type))

    if config.pretrained:
        print("Downloading pretrained model:", flush=True)
        state_dict = load_state_dict_from_url(
            model_urls[config.type], progress=True
        )
        model.load_state_dict(state_dict)

    # Add wrappers
    if config.get_intermediate_rep:
        model = IntermediateLayerGetter(
            model, return_layers=config.get_intermediate_rep, keep_output=True
        )
    if config.noise_adv_regression or config.noise_adv_classification:
        assert not config.self_attention
        model = NoiseAdvWrapper(
            model,
            input_size=model.fc.in_features if "resnet" in config.type else model.n_features,
            hidden_size=model.fc.in_features if "resnet" in config.type else 4096,
            classification=config.noise_adv_classification,
        )
    print("Model with {} parameters.".format(get_model_parameters(model)))
    return model
