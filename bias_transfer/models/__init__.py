import torch
import numpy as np

from .resnet import ResNet, Bottleneck, BasicBlock
from .noise_adversarial_resnet import NoiseAdvResNet


def resnet_builder(data_loader,
                   seed: int,
                   type: int,
                   num_classes: int,
                   noise_adv_classification: bool,
                   noise_adv_regression: bool,
                   *args,
                   **kwargs):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if type in (18, 34):
        block = BasicBlock
    else:
        block = Bottleneck
    if type == 18:
        num_blocks = [2, 2, 2, 2]
    elif type == 34:
        num_blocks = [3, 4, 6, 3]
    elif type == 50:
        num_blocks = [3, 4, 6, 3]
    elif type == 101:
        num_blocks = [3, 4, 23, 3]
    elif type == 152:
        num_blocks = [3, 8, 36, 3]
    else:
        raise KeyError
    if noise_adv_regression or noise_adv_classification:
        return NoiseAdvResNet(block, num_blocks, num_classes=num_classes, classification=noise_adv_classification)
    else:
        return ResNet(block, num_blocks, num_classes=num_classes)

