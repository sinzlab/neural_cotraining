import numpy as np
import torch
from torch import nn
from functools import partial

from .main_loop_module import MainLoopModule


class RandomReadoutReset(MainLoopModule):
    def __init__(self, config, device, data_loader, seed):
        super(RandomReadoutReset, self).__init__(config, device, data_loader, seed)
        self.batch_progress = 0
        self.epoch_progress = 0

    def pre_epoch(self, model, train_mode):
        if train_mode and self.config.reset_linear_frequency.get("epoch"):
            if self.epoch_progress % self.config.reset_linear_frequency["epoch"] == 0:
                model.module.linear_readout.reset_parameters()
            self.epoch_progress += 1

    def pre_forward(self, model, inputs, shared_memory, train_mode):
        if train_mode and self.config.reset_linear_frequency.get("batch"):
            if self.batch_progress % self.config.reset_linear_frequency["batch"] == 0:
                model.module.linear_readout.reset_parameters()
            self.batch_progress += 1
        return model, inputs
