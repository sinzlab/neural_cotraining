import unittest
from bias_transfer.configs import trainer
from bias_transfer.tests._base import BaseTest


class MinimalTrainingTest(BaseTest):
    def test_training_adaptive_lr_schedule(self):
        print("===================================================", flush=True)
        print("=========TEST adaptive_lr training=================", flush=True)
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
        print("===================================================", flush=True)
        print("===========TEST fixed_lr training==================", flush=True)
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
