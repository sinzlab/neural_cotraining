import unittest
import os
import torch
import numpy as np
import nnfabrik as nnf
from bias_transfer.configs import trainer
from bias_transfer.tests.base import BaseTest
from bias_transfer.utils import weight_reset


class MinimalTrainingTest(BaseTest):
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
