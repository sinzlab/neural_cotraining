import torch
import numpy as np
from torchvision.transforms import transforms

from .main_loop_module import MainLoopModule


class NoiseAugmentation(MainLoopModule):
    def __init__(self, model, config, device, data_loader, seed):
        super().__init__(model, config, device, data_loader, seed)
        self.rnd_gen = None
        self.img_min = 0
        self.img_max = 1
        self.noise_scale = None

    @staticmethod
    def apply_noise(
        x,
        device,
        std: dict = None,
        snr: dict = None,
        rnd_gen=None,
        img_min=0,
        img_max=1,
        noise_scale=None,
    ):
        if len(x.shape) == 3:
            x.unsqueeze(0)  # if we only have a single element

        with torch.no_grad():
            if std:
                noise_levels = std
            elif snr:
                noise_levels = snr
            else:
                noise_levels = {-1: 1.0}
            assert (
                abs(sum(noise_levels.values()) - 1.0) < 0.00001
            ), "Percentage for noise levels should sum to one!"
            indices = torch.randperm(x.shape[0])
            applied_std = torch.zeros([x.shape[0], 1], device=device, dtype=torch.float32)
            start = 0
            for (
                level,
                percentage,
            ) in noise_levels.items():  # TODO: is this efficient enough?
                end = start + int(percentage * x.shape[0])
                if isinstance(level, tuple):  # select level randomly from range
                    level = torch.empty(1, device=device, dtype=torch.float32).uniform_(
                        level[0], level[1]
                    )
                if level > 0:  # select specified noise level for a fraction of the data
                    if std is None:  # are we doing snr or std?
                        signal = torch.mean(
                            x[indices[start:end]] * x[indices[start:end]],
                            dim=[1, 2, 3],
                            keepdim=True,
                        )  # for each dimension except batch
                        std = signal / level
                    else:
                        if not isinstance(level, torch.Tensor):
                            std = torch.tensor(level, device=device)
                        else:
                            std = level
                    std = std.float()
                    applied_std[indices[start:end]] = std.squeeze().unsqueeze(-1)
                    std = std.expand_as(x[start:end])
                    noise = torch.normal(mean=0.0, std=std, generator=rnd_gen)
                    if noise_scale is not None:
                        noise *= noise_scale
                    x[indices[start:end]] += noise
                # else: deactivate noise for a fraction of the data
                start = end
            if isinstance(img_max, torch.Tensor):
                for i in range(
                    img_max.shape[0]
                ):  # clamp each color channel individually
                    x[:, i] = torch.clamp(x[:, i], max=img_max[i], min=img_min[i])
            else:
                x = torch.clamp(x, max=img_max, min=img_min)
        return x, applied_std

