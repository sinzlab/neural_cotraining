import unittest
import os
import torch
import numpy as np
import nnfabrik as nnf
from bias_transfer.configs import trainer, model
from bias_transfer.models.utils import weight_reset
from bias_transfer.tests._base import BaseTest


class TransferTest(BaseTest):
    def test_transfer_training(self):
        print("===================================================", flush=True)
        print("============TEST transfer training=================", flush=True)
        pretrained_path = "./checkpoint/ckpt.{}.pth".format(
            nnf.utility.dj_helpers.make_hash("test1")
        )
        transfer_path = "./checkpoint/ckpt.to_transfer.pth"
        if os.path.exists(transfer_path):
            os.remove(transfer_path)
        pretrain_trainer_conf = trainer.TrainerConfig(
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
        score = self.run_training(pretrain_trainer_conf)
        self.assertAlmostEqual(score, 46.76, places=1)
        os.rename(pretrained_path, transfer_path)
        transfer_trainer_conf = trainer.TrainerConfig(
            comment="Transfer Training Test",
            max_iter=2,
            verbose=False,
            add_noise=False,
            noise_snr=None,
            noise_std=None,
            noise_test={"noise_snr": [], "noise_std": [],},
            restore_best=False,
            lr_milestones=(1,),
            adaptive_lr=False,
            patience=1000,
            transfer_from_path=transfer_path,
            freeze=("core",),
            reset_linear=True,
        )
        score = self.run_training(transfer_trainer_conf)
        self.assertAlmostEqual(score, 9.76, places=1)

    def test_rdm_transfer_training(self):
        print("===================================================", flush=True)
        print("===========TEST RDM transfer training==============", flush=True)
        pretrained_path = "./checkpoint/ckpt.{}.pth".format(
            nnf.utility.dj_helpers.make_hash("test1")
        )
        transfer_path = "./checkpoint/ckpt.to_transfer.pth"
        if os.path.exists(transfer_path):
            os.remove(transfer_path)
        pretrain_trainer_conf = trainer.TrainerConfig(
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
        score = self.run_training(pretrain_trainer_conf)
        self.assertAlmostEqual(score, 46.76, places=1)
        os.rename(pretrained_path, transfer_path)
        model_conf = model.ClassificationModelConfig(
            comment="CIFAR10",
            dataset_cls="CIFAR10",
            type="resnet18",
            rdm_prediction=True
        )
        self.get_parts(self.dataset_conf, model_conf, self.seed)
        transfer_trainer_conf = trainer.TrainerConfig(
            comment="RDM Transfer Training Test",
            max_iter=2,
            verbose=False,
            add_noise=False,
            noise_snr=None,
            noise_std=None,
            noise_test={"noise_snr": [], "noise_std": [],},
            restore_best=False,
            lr_milestones=(1,),
            adaptive_lr=False,
            patience=1000,
            transfer_from_path=transfer_path,
            # freeze=("core",),
            # reset_linear=True,
            rdm_transfer=True,
            rdm_prediction={"lambda": 1.0},
        )
        score = self.run_training(transfer_trainer_conf)
        self.assertAlmostEqual(score, 9.88, places=1)
        # reset model
        self.setUpClass()

    def test_transfer_training_vgg(self):
        print("===================================================", flush=True)
        print("=========TEST transfer training (VGG)==============", flush=True)
        model_conf = model.ClassificationModelConfig(
            comment="CIFAR10",
            dataset_cls="CIFAR10",
            type="vgg19_bn",
        )
        self.get_parts(self.dataset_conf, model_conf, self.seed)
        pretrained_path = "./checkpoint/ckpt.{}.pth".format(
            nnf.utility.dj_helpers.make_hash("test1")
        )
        transfer_path = "./checkpoint/ckpt.to_transfer.pth"
        if os.path.exists(transfer_path):
            os.remove(transfer_path)
        pretrain_trainer_conf = trainer.TrainerConfig(
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
            readout_name="classifier"
        )
        score = self.run_training(pretrain_trainer_conf)
        self.assertAlmostEqual(score, 17.92, places=1)
        os.rename(pretrained_path, transfer_path)
        transfer_trainer_conf = trainer.TrainerConfig(
            comment="Transfer Training Test",
            max_iter=2,
            verbose=False,
            add_noise=False,
            noise_snr=None,
            noise_std=None,
            noise_test={"noise_snr": [], "noise_std": [],},
            restore_best=False,
            lr_milestones=(1,),
            adaptive_lr=False,
            patience=1000,
            transfer_from_path=transfer_path,
            freeze=("core",),
            reset_linear=True,
            readout_name="classifier"
        )
        score = self.run_training(transfer_trainer_conf)
        self.assertAlmostEqual(score, 9.6, places=1)
        self.setUpClass()

    def test_rdm_transfer_training_vgg(self):
        print("===================================================", flush=True)
        print("=======TEST RDM transfer training (VGG)============", flush=True)
        model_conf = model.ClassificationModelConfig(
            comment="CIFAR10",
            dataset_cls="CIFAR10",
            type="vgg19_bn",
        )
        self.get_parts(self.dataset_conf, model_conf, self.seed)
        pretrained_path = "./checkpoint/ckpt.{}.pth".format(
            nnf.utility.dj_helpers.make_hash("test1")
        )
        transfer_path = "./checkpoint/ckpt.to_transfer.pth"
        if os.path.exists(transfer_path):
            os.remove(transfer_path)
        pretrain_trainer_conf = trainer.TrainerConfig(
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
            readout_name="classifier"
        )
        score = self.run_training(pretrain_trainer_conf)
        self.assertAlmostEqual(score, 17.92, places=1)
        os.rename(pretrained_path, transfer_path)
        model_conf = model.ClassificationModelConfig(
            comment="CIFAR10",
            dataset_cls="CIFAR10",
            type="vgg19_bn",
            rdm_prediction=True
        )
        self.get_parts(self.dataset_conf, model_conf, self.seed)
        transfer_trainer_conf = trainer.TrainerConfig(
            comment="RDM Transfer Training Test",
            max_iter=2,
            verbose=False,
            add_noise=False,
            noise_snr=None,
            noise_std=None,
            noise_test={"noise_snr": [], "noise_std": [],},
            restore_best=False,
            lr_milestones=(1,),
            adaptive_lr=False,
            patience=1000,
            transfer_from_path=transfer_path,
            rdm_transfer=True,
            rdm_prediction={"lambda": 1.0},
            readout_name="classifier"
        )
        score = self.run_training(transfer_trainer_conf)
        self.assertAlmostEqual(score, 10.2, places=1)
        # reset model
        self.setUpClass()

if __name__ == "__main__":
    unittest.main()
