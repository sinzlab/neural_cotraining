from nnvision.models.models import se_core_gauss_readout, se_core_point_readout
from neural_cotraining.models.utils import get_model_parameters

def neural_cnn_builder(data_loaders, seed: int = 1000, **config):
    config.pop("comment", None)
    readout_type = config.pop("readout_type", None)
    if readout_type == "point":
        model = se_core_point_readout(dataloaders=data_loaders, seed=seed, **config)
    elif readout_type == "gauss":
        model = se_core_gauss_readout(dataloaders=data_loaders, seed=seed, **config)
    print("Model with {} parameters.".format(get_model_parameters(model)))
    return model