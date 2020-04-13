from torch import nn
import torch

from .main_loop_module import MainLoopModule


def compute_corr_matrix(x):
    x_flat = x.flatten(1, -1)
    centered = x_flat - x_flat.mean(dim=1).view(-1, 1)
    result = (centered @ centered.transpose(0, 1)) / torch.ger(
        torch.norm(centered, 2, dim=1), torch.norm(centered, 2, dim=1)
    )  # see https://de.mathworks.com/help/images/ref/corr2.html
    return result


def compute_cosine_matrix(x):
    x_flat = x.flatten(1, -1)
    centered = x_flat - x_flat.mean(dim=0).view(-1, 1)  # centered by mean over images
    result = (centered @ centered.transpose(0, 1)) / torch.ger(
        torch.norm(centered, 2, dim=1), torch.norm(centered, 2, dim=1)
    )  # see https://de.mathworks.com/help/images/ref/corr2.html
    return result


def arctanh(x):
    return 0.5 * torch.log((1 + x) / (1 - x))


class RDMPrediction(MainLoopModule):
    def __init__(self, config, device, data_loader, seed):
        super().__init__(config, device, data_loader, seed)
        # self.rep = self.config.representation_matching.get("representation", "conv_rep")
        self.criterion = nn.MSELoss()

    def pre_forward(self, model, inputs, shared_memory, train_mode):
        return model, inputs[0]

    def post_forward(self, outputs, loss, targets, extra_losses, applied_std, **kwargs):
        pred_rdm = arctanh(compute_cosine_matrix(outputs["conv_rep"]))
        trg_rdm = arctanh(compute_cosine_matrix(targets[1]))

        pred_loss = self.criterion(pred_rdm, trg_rdm)
        loss += self.config.rdm_prediction.get("lambda", 1.0) * pred_loss
        extra_losses["RDMPrediction"] += pred_loss.item()
        return outputs, loss, targets[0]
