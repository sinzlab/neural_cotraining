import numpy as np
import torch
from torch.autograd import Variable
import torchvision
import torch.nn as nn
from torchvision.models import vgg
from torchvision.models.vgg import VGG as DefaultVGG


def create_vgg_readout(readout_type, n_features, num_classes=None):
    if readout_type == "dense":
        readout = nn.Sequential(
            nn.Linear(n_features, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    elif readout_type == "conv":
        readout = nn.Sequential(
            nn.Conv2d(512, 4096, 1, 1, bias=True),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, 1, 1, bias=True),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Conv2d(4096, num_classes, 1, 1, bias=True),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
        )
    return readout


class VGG(DefaultVGG):
    def __init__(
        self,
        cfg,
        batch_norm=False,
        input_size=64,
        num_classes=200,
        avg_pool=False,
        readout_type="dense",
        init_weights=True,
        input_channels=3,
    ):
        nn.Module.__init__(self)
        self.input_channels = input_channels
        self.features = vgg.make_layers(vgg.cfgs[cfg], batch_norm=batch_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7)) if avg_pool else None
        self.readout_type = readout_type

        test_input = Variable(torch.zeros(1, 3, input_size, input_size))
        test_out = self.features(test_input)
        self.n_features = test_out.size(1) * test_out.size(2) * test_out.size(3)
        self.classifier = create_vgg_readout(
            readout_type, n_features=self.n_features, num_classes=num_classes
        )
        self.flatten = nn.Flatten(start_dim=1)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        if self.input_channels == 1:
            x = x.expand(-1, 3, -1, -1)
        x = self.features(x)
        if self.avgpool:
            x = self.avgpool(x)
        if self.readout_type == "dense":
            # core_out = core_out.view(core_out.size(0), -1)
            x = self.flatten(x)
        x = self.classifier(x)
        return x


def vgg_builder(seed: int, config):
    if "11" in config.type:
        cfg = "A"
    elif "13" in config.type:
        cfg = "B"
    elif "16" in config.type:
        cfg = "D"
    elif "19" in config.type:
        cfg = "E"
    else:
        raise NameError("Unknown VGG Type")

    model = VGG(
        cfg=cfg,
        batch_norm="bn" in config.type,
        input_size=config.input_size,
        num_classes=config.num_classes,
        avg_pool=config.avg_pool,
        readout_type=config.readout_type,
        input_channels=config.input_channels,
    )
    return model
