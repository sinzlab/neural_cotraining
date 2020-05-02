import os
import unittest

import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler
from torch.utils.data.sampler import SequentialSampler

import nnfabrik as nnf
from bias_transfer.configs import dataset, model, trainer
from bias_transfer.models.utils import weight_reset


class BaseTest(unittest.TestCase):
    dataset_conf = dataset.ImageDatasetConfig(
        comment="Minimal CIFAR10",
        dataset_cls="CIFAR10",
        apply_data_normalization=False,
        apply_data_augmentation=False,
        add_corrupted_test=False,
        valid_size=0.95,
    )
    model_conf = model.ClassificationModelConfig(
        comment="CIFAR10 ResNet18", dataset_cls="CIFAR10", type="18"
    )
    seed = 42

    @classmethod
    def run_training(cls, trainer_conf):
        uid = "test1"
        path = "./checkpoint/ckpt.{}.pth".format(nnf.utility.dj_helpers.make_hash(uid))
        if os.path.exists(path):
            os.remove(path)
        torch.manual_seed(cls.seed)
        np.random.seed(cls.seed)
        torch.cuda.manual_seed(cls.seed)
        cls.model.apply(weight_reset)

        trainer_fn = nnf.builder.get_trainer(trainer_conf.fn, trainer_conf.to_dict())

        def call_back(**kwargs):
            pass

        # model training
        score, output, model_state = trainer_fn(
            model=cls.model,
            dataloaders=cls.data_loaders,
            seed=cls.seed,
            uid=uid,
            cb=call_back,
        )
        return score

    @classmethod
    def get_parts(cls, dataset_conf, model_conf, seed):
        os.chdir("/work/")
        cls.data_loaders, cls.model = nnf.builder.get_all_parts(
            dataset_fn=dataset_conf.fn,
            dataset_config=dataset_conf.to_dict(),
            model_fn=model_conf.fn,
            model_config=model_conf.to_dict(),
            seed=seed,
            trainer_fn=None,
            trainer_config=None,
        )
        cls.data_loaders["validation"] = cls.data_loaders["train"]
        cls.data_loaders["test"] = cls.data_loaders["train"]

    @classmethod
    def setUpClass(cls):  # called once before all methods of the class
        cls.get_parts(cls.dataset_conf, cls.model_conf, cls.seed)

