import os
import unittest

import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler

import nnfabrik as nnf
from bias_transfer.configs import dataset, model, trainer
from bias_transfer.utils import weight_reset


class BaseTest(unittest.TestCase):
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
    def setUpClass(cls):  # called once before all methods of the class
        os.chdir("/work/")
        cls.dataset_conf = dataset.ImageDatasetConfig(
            comment="Minimal CIFAR10",
            dataset_cls="CIFAR10",
            apply_data_normalization=False,
            add_corrupted_test=False,
            valid_size=0.05,
        )
        cls.model_conf = model.ClassificationModelConfig(
            comment="CIFAR10 ResNet18", dataset_cls="CIFAR10", type="18"
        )
        cls.trainer_conf = trainer.TrainerConfig(comment="Basic Trainer")
        cls.seed = 42

        cls.data_loaders, cls.model = nnf.builder.get_all_parts(
            dataset_fn=cls.dataset_conf.fn,
            dataset_config=cls.dataset_conf.to_dict(),
            model_fn=cls.model_conf.fn,
            model_config=cls.model_conf.to_dict(),
            seed=cls.seed,
            trainer_fn=None,
            trainer_config=None,
        )
        train = cls.data_loaders["train"]
        sampler = SubsetRandomSampler(train.sampler.indices[:800])
        cls.data_loaders["train"] = torch.utils.data.DataLoader(
            train.dataset,
            batch_size=train.batch_size,
            sampler=sampler,
            num_workers=train.num_workers,
            pin_memory=train.pin_memory,
            shuffle=False,
        )