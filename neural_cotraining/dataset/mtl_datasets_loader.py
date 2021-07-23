from .img_classification_loader import ImageClassificationLoader
from .neural_dataset_loader import NeuralDatasetLoader


class MTLDatasetsLoader:
    def __call__(self, seed, **config):
        seed = 1000
        v1_dataset_config = config.pop("v1_dataset_config")
        v4_dataset_config = config.pop("v4_dataset_config")
        img_dataset_config = config.pop("img_dataset_config")
        classification_loader = config.pop("classification_loader")

        data_loaders = {"train": {}, "validation": {}, "test": {}}
        if v1_dataset_config:
            v1_dataset_config.pop("seed")
            v1_dataset_loaders = NeuralDatasetLoader()(seed, **v1_dataset_config)
            data_loaders["train"]["v1"] = v1_dataset_loaders["train"]["v1"]
            data_loaders["validation"]["v1"] = v1_dataset_loaders["validation"]["v1"]
            data_loaders["test"]["v1"] = v1_dataset_loaders["test"]["v1"]
        if v4_dataset_config:
            v4_dataset_config.pop("seed")
            v4_dataset_loaders = NeuralDatasetLoader()(seed, **v4_dataset_config)
            data_loaders["train"]["v4"] = v4_dataset_loaders["train"]["v4"]
            data_loaders["validation"]["v4"] = v4_dataset_loaders["validation"]["v4"]
            data_loaders["test"]["v4"] = v4_dataset_loaders["test"]["v4"]

        if classification_loader == "img_classification":
            img_dataset_loaders = ImageClassificationLoader()(
                seed, **img_dataset_config
            )
            data_loaders["train"]["img_classification"] = img_dataset_loaders["train"][
                "img_classification"
            ]
        else:
            img_dataset_config.pop("seed")
            img_dataset_loaders = NeuralDatasetLoader()(seed, **img_dataset_config)
            data_loaders["train"]["img_classification"] = img_dataset_loaders["train"]
        data_loaders["validation"]["img_classification"] = img_dataset_loaders[
            "validation"
        ]["img_classification"]
        data_loaders["test"]["img_classification"] = img_dataset_loaders["test"][
            "img_classification"
        ]
        if "test_out_domain" in img_dataset_loaders.keys():
            data_loaders["test_out_domain"] = {}
            data_loaders["validation_out_domain"] = {}
            data_loaders["test_out_domain"]["img_classification"] = img_dataset_loaders[
                "test_out_domain"
            ]["img_classification"]
            data_loaders["validation_out_domain"][
                "img_classification"
            ] = img_dataset_loaders["validation_out_domain"]["img_classification"]
        if "validation_gauss" in img_dataset_loaders.keys():
            data_loaders["validation_gauss"] = img_dataset_loaders["validation_gauss"]
        if "fly_c_test" in img_dataset_loaders:
            data_loaders["fly_c_test"] = img_dataset_loaders["fly_c_test"]
        if "imagenet_fly_c_test" in img_dataset_loaders:
            data_loaders["imagenet_fly_c_test"] = img_dataset_loaders[
                "imagenet_fly_c_test"
            ]
        if "c_test" in img_dataset_loaders:
            data_loaders["c_test"] = img_dataset_loaders["c_test"]
        if "st_test" in img_dataset_loaders:
            data_loaders["st_test"] = img_dataset_loaders["st_test"]
        return data_loaders


mtl_loader = MTLDatasetsLoader()
