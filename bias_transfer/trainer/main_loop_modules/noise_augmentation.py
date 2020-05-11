import torch
import numpy as np
from torchvision.transforms import transforms

from .main_loop_module import MainLoopModule


class NoiseAugmentation(MainLoopModule):
    def __init__(self, model, config, device, data_loader, seed):
        super().__init__(model, config, device, data_loader, seed)
        self.rnd_gen = None
        if isinstance(data_loader, dict):
            train_loader = data_loader["img_classification"]
        else:
            train_loader = data_loader.loaders["img_classification"]
        transform_list = train_loader.dataset.transforms.transform.transforms
        # go through StandardTransform and Compose to get to  the actual transforms
        normalization = None
        for trans in transform_list:
            if isinstance(trans, transforms.Normalize):
                normalization = trans
        if normalization:
            # image = (image - mean) / std
            # => noisy_image = (image + noise - mean) / std
            #                = (image - mean) / std + noise/ std
            img_mean = torch.tensor(normalization.mean)
            img_std = torch.tensor(normalization.std)
            self.img_min = -img_mean / img_std
            self.img_max = (1 - img_mean) / img_std
            self.noise_scale = 1 / img_std
            self.noise_scale = self.noise_scale.view(1, -1, 1, 1).to(device)
        else:
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
            applied_std = torch.zeros([x.shape[0], 1], device=device)
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

    def pre_epoch(self, model, train_mode, epoch, **kwargs):
        if not train_mode:
            rnd_gen = torch.Generator(device=self.device)
            if isinstance(self.seed, np.generic):
                self.seed = np.asscalar(self.seed)
            self.rnd_gen = rnd_gen.manual_seed(
                self.seed
            )  # so that we always have the same noise for evaluation!

    def pre_forward(self, model, inputs, shared_memory, train_mode, **kwargs):
        inputs, shared_memory["applied_std"] = self.apply_noise(
            inputs,
            self.device,
            std=self.config.noise_std,
            snr=self.config.noise_snr,
            rnd_gen=self.rnd_gen if not train_mode else None,
            img_min=self.img_min,
            img_max=self.img_max,
            noise_scale=self.noise_scale,
        )
        return model, inputs
