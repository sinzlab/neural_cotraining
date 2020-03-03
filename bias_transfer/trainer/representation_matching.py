from torch import nn
import torch

from .noise_augmentation import NoiseAugmentation


class RepresentationMatching(NoiseAugmentation):
    def __init__(self, config, device, data_loader, seed):
        super().__init__(config, device, data_loader, seed)
        self.rep = self.config.representation_matching.get("representation", "conv_rep")
        if self.config.representation_matching.get("criterion", "cosine") == "cosine":
            self.criterion = nn.CosineEmbeddingLoss()
        else:
            self.criterion = nn.MSELoss()

    def pre_forward(self, model, inputs, shared_memory, train_mode):
        model, inputs1 = super().pre_forward(model, inputs, shared_memory, train_mode)
        if self.config.representation_matching.get("second_noise_std", None) \
                or self.config.representation_matching.get("second_noise_snr", None):
            inputs2, _ = self.apply_noise(inputs,
                                       self.device,
                                       std=self.config.representation_matching.get("second_noise_std", None),
                                       snr=self.config.representation_matching.get("second_noise_snr", None),
                                       rnd_gen=self.rnd_gen if not train_mode else None)
        else:
            inputs2 = inputs
        inputs = torch.cat([inputs1, inputs2])
        return model, inputs

    def post_forward(self, outputs, loss, extra_losses, **kwargs):
        rep_1 = outputs[self.rep][:(outputs[self.rep].shape[0] // 2)]
        rep_2 = outputs[self.rep][(outputs[self.rep].shape[0] // 2):]
        o = torch.ones(rep_1.shape[:1], device=self.device)
        sim_loss = self.criterion(rep_1, rep_2,
                                  o)  # ones indicating that we want to measure similarity
        loss += self.config.representation_matching.get("lambda", 1.0) * sim_loss
        outputs["logits"] = outputs["logits"][:(outputs["logits"].shape[0] // 2)]
        extra_losses["RepresentationMatching"] += sim_loss.item()
        return outputs, loss
