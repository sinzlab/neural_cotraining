from .img_dataset_loader import img_dataset_loader
from .neural_dataset_loader import neural_dataset_loader


def mtl_datasets_loader(seed, **config):
    neural_dataset_config = config.pop("neural_dataset_config", None)
    img_dataset_config = config.pop("img_dataset_config", None)

    neural_dataset_config.pop("seed")

    neural_dataset_loaders = neural_dataset_loader(seed, **neural_dataset_config)
    img_dataset_loaders = img_dataset_loader(seed, **img_dataset_config)
    neural_dataset_loaders["train"]["img_classification"] = img_dataset_loaders[
        "train"
    ]["img_classification"]
    data_loaders = {
        "train": neural_dataset_loaders["train"],
        "validation": {
            "img_classification": img_dataset_loaders["validation"][
                "img_classification"
            ],
            "neural": neural_dataset_loaders["validation"]["neural"],
        },
        "test": {
            "img_classification": img_dataset_loaders["test"]["img_classification"],
            "neural": neural_dataset_loaders["test"]["neural"],
        },
    }
    if "c_test" in img_dataset_loaders:
        data_loaders["c_test"] = img_dataset_loaders["c_test"]
    return data_loaders
