import unittest
import os
import torch
import numpy as np
import nnfabrik as nnf
from torch.utils.data.sampler import SubsetRandomSampler
from bias_transfer.configs import model, dataset, trainer
from bias_transfer.utils import weight_reset


class MinimalTrainingTest(unittest.TestCase):
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

    def test_training_adaptive_lr_schedule(self):
        trainer_conf = trainer.TrainerConfig(
            comment="Minimal Training Test",
            max_iter=3,
            verbose=False,
            noise_test={"noise_snr": [], "noise_std": [],},
            restore_best=False,
            lr_milestones=None,
            adaptive_lr=True,
            patience=2,
        )
        score = self.run_training(trainer_conf)
        self.assertAlmostEqual(score, 11.56)  # before the merge: 13.52

    def test_training_fixed_lr_schedule(self):
        trainer_conf = trainer.TrainerConfig(
            comment="Minimal Training Test",
            max_iter=3,
            verbose=False,
            noise_test={"noise_snr": [], "noise_std": [],},
            restore_best=False,
            lr_milestones=(1, 2),
            adaptive_lr=False,
            patience=1000,
        )
        score = self.run_training(trainer_conf)
        self.assertAlmostEqual(score, 10.0)  # before the merge: 12.04


if __name__ == "__main__":
    unittest.main()
