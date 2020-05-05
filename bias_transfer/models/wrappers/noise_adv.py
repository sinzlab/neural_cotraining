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
        input_size,
        hidden_size,
        classification: bool = False,
        num_noise_readout_layers: int = 1,
    ):
        super().__init__()
        self.model = model

        noise_readout_layers = []
        for i in range(0, num_noise_readout_layers):
            in_size = input_size if i == 0 else hidden_size
            out_size = 1 if i == num_noise_readout_layers-1 else hidden_size
            noise_readout_layers.append(nn.Linear(in_size, out_size))
            if i < num_noise_readout_layers-1:
                noise_readout_layers.append(nn.ReLU())
        self.noise_readout = nn.Sequential(*noise_readout_layers)
        self.nonlinearity = nn.Sigmoid() if classification else nn.ReLU()

    def forward(self, x, seed: int = None, noise_lambda=None):
        extra_output, out = self.model(x)
        core_out = extra_output["core"]
        noise_out = self.noise_readout(grad_reverse(core_out, noise_lambda))
        noise_out = self.nonlinearity(noise_out)
        extra_output["noise_pred"] = noise_out
        return extra_output, out
