import unittest
import torch
import numpy as np
from bias_transfer.configs import trainer
from bias_transfer.models.utils import weight_reset
from bias_transfer.tests._base import BaseTest


class TrainingTest(BaseTest):
    def test_training_adaptive_lr_schedule(self):
        print("===================================================", flush=True)
        print("=========TEST adaptive_lr training=================", flush=True)
        trainer_conf = trainer.TrainerConfig(
            comment="Adaptive LR Training Test",
            max_iter=3,
            verbose=False,
            noise_test={"noise_snr": [], "noise_std": [],},
            restore_best=False,
            lr_milestones=None,
            adaptive_lr=True,
            patience=2,
        )
        score = self.run_training(trainer_conf)
        self.assertAlmostEqual(score, 77.1, places=1)

    def test_training_fixed_lr_schedule(self):
        print("===================================================", flush=True)
        print("===========TEST fixed_lr training==================", flush=True)
        trainer_conf = trainer.TrainerConfig(
            comment="Fixed LR Training Test",
            max_iter=3,
            verbose=False,
            noise_test={"noise_snr": [], "noise_std": [],},
            restore_best=False,
            lr_milestones=(1, 2),
            adaptive_lr=False,
            patience=1000,
        )
        score = self.run_training(trainer_conf)
        self.assertAlmostEqual(score, 62.6, places=1)

    def test_training_noise_augment_std(self):
        print("===================================================", flush=True)
        print("==========TEST noise-augmented training STD =======", flush=True)
        trainer_conf = trainer.TrainerConfig(
            comment="Noise Augmented Training Test",
            max_iter=2,
            verbose=False,
            add_noise=True,
            noise_snr=None,
            noise_std={0.08: 0.1, 0.12: 0.1, 0.18: 0.1, 0.26: 0.1, 0.38: 0.1, -1: 0.5,},
            noise_test={"noise_snr": [], "noise_std": [],},
            restore_best=False,
            lr_milestones=(1,),
            adaptive_lr=False,
            patience=1000,
        )
        score = self.run_training(trainer_conf)
        self.assertAlmostEqual(score, 44.7, places=1)

    def test_training_noise_augment_snr(self):
        print("===================================================", flush=True)
        print("===========TEST noise-augmented training SNR=======", flush=True)
        trainer_conf = trainer.TrainerConfig(
            comment="Noise Augmented Training Test",
            max_iter=2,
            verbose=False,
            add_noise=True,
            noise_snr={1.0: 0.25, 1.5: 0.25, -1: 0.5},
            noise_std=None,
            noise_test={"noise_snr": [], "noise_std": [],},
            restore_best=False,
            lr_milestones=(1,),
            adaptive_lr=False,
            patience=1000,
        )
        score = self.run_training(trainer_conf)
        self.assertAlmostEqual(score, 43.1, places=1)

    def test_freeze_params(self):
        print("===================================================", flush=True)
        print("=============TEST freeze params====================", flush=True)
        trainer_conf = trainer.TrainerConfig(
            comment="Fixed LR Training Test",
            max_iter=2,
            verbose=False,
            noise_test={"noise_snr": [], "noise_std": [],},
            restore_best=False,
            lr_milestones=(1,),
            adaptive_lr=False,
            freeze=("readout",),
            patience=1000,
        )
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        self.model.apply(weight_reset)
        readout_params_before = torch.clone(self.model.named_params()["fc"].data)
        _ = self.run_training(trainer_conf)
        readout_params_after = self.model.named_params()["fc"].data
        self.assertTrue(
            torch.all(torch.eq(readout_params_before, readout_params_after))
        )


if __name__ == "__main__":
    unittest.main()
