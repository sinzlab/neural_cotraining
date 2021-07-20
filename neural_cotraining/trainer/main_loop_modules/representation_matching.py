from torch import nn
import torch

from .noise_augmentation import NoiseAugmentation


class RepresentationMatching(NoiseAugmentation):
    def __init__(self, model, config, device, data_loader, seed):
        super().__init__(model, config, device, data_loader, seed)
        self.rep = self.config.representation_matching.get("representation", "conv_rep")
        if self.config.representation_matching.get("criterion", "cosine") == "cosine":
            self.criterion = nn.CosineEmbeddingLoss()
        else:
            self.criterion = nn.MSELoss()


    def post_forward(self, outputs, loss, targets, extra_losses, train_mode, **kwargs):
        self.batch_size = targets['img_classification'].shape[0]
        extra_outputs, outputs = outputs[0], outputs[1]
        if train_mode:
            self.clean_flags = torch.ones((self.batch_size,)).type(torch.BoolTensor)
            rep_1 = extra_outputs[self.rep][: self.batch_size][self.clean_flags]
            rep_2 = extra_outputs[self.rep][self.batch_size :]
            if self.config.representation_matching.get("criterion", "cosine") == "cosine":
                o = torch.ones(
                    rep_1.shape[:1], device=self.device
                ).unsqueeze_(1).unsqueeze_(1)  # ones indicating that we want to measure similarity
                sim_loss = self.criterion(rep_1, rep_2, o)
            else:
                sim_loss = self.criterion(rep_1, rep_2)
            loss += self.config.representation_matching.get("lambda", 1.0) * sim_loss
            for k, v in extra_outputs.items():
                if isinstance(v, torch.Tensor):
                    extra_outputs[k] = v[: self.batch_size]
            extra_losses["RepresentationMatching"] += sim_loss.item()
        outputs = outputs[: self.batch_size]
        return (extra_outputs, outputs), loss, targets
