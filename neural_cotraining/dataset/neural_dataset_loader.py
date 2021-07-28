import numpy as np
import torch

from nnfabrik import builder
import os
from os import listdir
from os.path import isfile, join


class NeuralDatasetLoader:
    def __call__(self, seed, **config):
        seed = 1000
        config.pop("comment", None)
        config.pop("shuffle", None)
        config.pop("valid_size", None)

        data_dir = config.pop("data_dir", None)  # main directory of data
        sessions_dir = config.pop(
            "sessions_dir", None
        )  # path to the sessions of the neural dataset

        # prepare args for the monkey loader
        neuronal_data_path = os.path.join(data_dir, "{}/".format(sessions_dir))
        config["neuronal_data_files"] = [
            neuronal_data_path + f
            for f in listdir(neuronal_data_path)
            if (
                isfile(join(neuronal_data_path, f))
                and f != "CSRF19_V4_3653663964522.pickle"
            )
        ]
        if not config["individual_image_paths"]:
            config["image_cache_path"] = os.path.join(data_dir, "images/individual")
            config["original_image_cache_path"] = os.path.join(
                data_dir, "images/original/"
            )
        else:
            config["image_cache_path"] = os.path.join(
                data_dir, "images/individual_image_paths.pickle"
            )
            config["original_image_cache_path"] = os.path.join(
                data_dir, "images/individual_image_paths.pickle"
            )

        # pass the args to the monkey loader from nnvision
        torch.manual_seed(seed)
        np.random.seed(seed)
        dataset_fn = "nnvision.datasets.monkey_static_loader"
        data_loaders = builder.get_data(dataset_fn, config)
        dataloaders = {
            "validation": {
                task: data_loaders["validation"] for task in config["target_types"]
            },
            "test": {task: data_loaders["test"] for task in config["target_types"]},
        }
        if len(config["target_types"]) == 1:
            dataloaders["train"] = {
                task: data_loaders["train"] for task in config["target_types"]
            }
        else:
            neural_set = "v1" if "v1" in config["target_types"] else "v4"
            dataloaders["train"] = {neural_set: data_loaders["train"]}
        if "validation_gauss" in data_loaders.keys():
            dataloaders["validation_gauss"] = data_loaders["validation_gauss"]
        if "fly_c_test" in data_loaders.keys():
            dataloaders["fly_c_test"] = data_loaders["fly_c_test"]

        return dataloaders


neural_loader = NeuralDatasetLoader()
