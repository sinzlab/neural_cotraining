import numpy as np
import torch
from torch import nn
from functools import partial

from .main_loop_module import MainLoopModule


class NoiseAdvTraining(MainLoopModule):
    def __init__(self, config, device, data_loader, seed):
        super().__init__(config, device, data_loader, seed)
        self.progress = 0.0
        self.step_size = float(config.num_epochs * len(data_loader)) / data_loader.batch_size
        if config.noise_adv_regression:
            self.criterion = nn.MSELoss()
        else:  # config.noise_adv_classification
            self.criterion = nn.BCELoss()

    def pre_forward(self, model, inputs, shared_memory, train_mode):
        noise_adv_lambda = 2. / (1. + np.exp(-self.config.noise_adv_gamma * self.progress)) - 1
        if train_mode:
            self.progress += self.step_size
        return partial(model, noise_lambda=noise_adv_lambda), inputs

    def post_forward(self, outputs, loss, extra_losses, applied_std=None, **kwargs):
        if applied_std is None:
            applied_std = torch.zeros_like(outputs["noise_pred"], device=self.device)
        if self.config.noise_adv_classification:
            applied_std = (applied_std > 0.0).type(torch.FloatTensor).to(device=self.device)
        noise_loss = self.criterion(outputs["noise_pred"], applied_std)
        extra_losses["NoiseAdvTraining"] += noise_loss.item()
        loss += self.config.noise_adv_loss_factor * noise_loss
        return outputs, loss
