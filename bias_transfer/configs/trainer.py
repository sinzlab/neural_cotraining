import copy
from functools import partial
from .base import BaseConfig
from nnfabrik.main import *


class TrainerConfig(BaseConfig):
    config_name = "trainer"
    table = Trainer()
    fn = "bias_transfer.trainer.trainer"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.force_cpu = kwargs.pop("force_cpu", False)
        self.use_tensorboard = kwargs.pop("use_tensorboard", False)
        self.optimizer = kwargs.pop("optimizer", "Adam")
        self.lr = kwargs.pop("lr", 0.0003)
        self.lr_decay = kwargs.pop("lr_decay", 0.8)
        self.weight_decay = kwargs.pop("weight_decay", 5e-4)
        self.momentum = kwargs.pop("momentum", 0.9)
        self.num_epochs = kwargs.pop("num_epochs", 200)
        self.lr_milestones = kwargs.pop("lr_milestones", (60, 120, 160))
        # noise
        self.add_noise = kwargs.pop("add_noise", False)
        self.noise_std = kwargs.pop("noise_std", None)
        self.noise_snr = kwargs.pop("noise_snr", None)
        self.noise_test = kwargs.pop("noise_test", {
            "noise_snr": [{5.0: 1.0}, {4.0: 1.0}, {3.0: 1.0}, {2.0: 1.0}, {1.0: 1.0}, {0.5: 1.0}, {0.0: 1.0}],
            "noise_std": [{0.0: 1.0}, {0.05: 1.0}, {0.1: 1.0}, {0.2: 1.0}, {0.3: 1.0}, {0.5: 1.0}, {1.0: 1.0}]
        })
        self.noise_adv_classification = kwargs.pop("noise_adv_classification", False)
        self.noise_adv_regression = kwargs.pop("noise_adv_regression", False)
        if self.noise_adv_classification or self.noise_adv_regression:
            self.main_loop_modules.append("NoiseAdvTraining")
        self.noise_adv_loss_factor = kwargs.pop("noise_adv_loss_factor", 1.0)
        self.noise_adv_gamma = kwargs.pop("noise_adv_gamma", 10.0)
        self.representation_matching = kwargs.pop("representation_matching", None)
        # transfer
        self.freeze = kwargs.pop("freeze", None)
        self.reset_linear = kwargs.pop("reset_linear", False)
        self.reset_linear_frequency = kwargs.pop("reset_linear_frequency", None)
        self.transfer_from_path = kwargs.pop("transfer_from_path", None)
        self.update(**kwargs)

    @property
    def main_loop_modules(self):
        modules = []
        if self.representation_matching:
            modules.append("RepresentationMatching")
        elif self.noise_snr or self.noise_std:  # Logit matching includes noise augmentation
            modules.append("NoiseAugmentation")
        if self.reset_linear_frequency:
            modules.append("RandomReadoutReset")
        return modules
