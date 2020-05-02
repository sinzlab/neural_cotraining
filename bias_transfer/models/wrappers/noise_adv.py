import torch.nn as nn
from torch.autograd import Function


# Used the implementation from https://github.com/CuthbertCai/pytorch_DANN
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambda_p):
        ctx.constant = lambda_p
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None


def grad_reverse(x, lambda_p):
    return GradReverse.apply(x, lambda_p)


class NoiseAdvWrapper(nn.Module):
    def __init__(
        self,
        model,
        readout_name: str,
        classification: bool = False,
        num_noise_readout_layers: int = 1,
    ):
        super().__init__()
        self.model = model
        readout_size = getattr(self.model._model, readout_name).in_features

        noise_readout_layers = []
        for i in range(1, num_noise_readout_layers):
            noise_readout_layers.append(nn.Linear(readout_size, readout_size))
            noise_readout_layers.append(nn.ReLU())
        noise_readout_layers.append(nn.Linear(readout_size, 1))
        self.noise_readout = nn.Sequential(*noise_readout_layers)
        self.classification = nn.Sigmoid() if classification else None

    def forward(self, x, seed: int = None, noise_lambda=None):
        extra_output, out = self.model(x)
        core_out = extra_output["core"]
        noise_out = self.noise_readout(grad_reverse(core_out, noise_lambda))
        if self.classification:
            noise_out = self.classification(noise_out)
        extra_output["noise_pred"] = noise_out
        return extra_output, out
