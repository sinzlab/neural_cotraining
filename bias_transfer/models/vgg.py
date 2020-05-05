import torch
from torch.autograd import Variable
import torchvision
import torch.nn as nn

VGG_TYPES = {
    "vgg11": torchvision.models.vgg11,
    "vgg11_bn": torchvision.models.vgg11_bn,
    "vgg13": torchvision.models.vgg13,
    "vgg13_bn": torchvision.models.vgg13_bn,
    "vgg16": torchvision.models.vgg16,
    "vgg16_bn": torchvision.models.vgg16_bn,
    "vgg19_bn": torchvision.models.vgg19_bn,
    "vgg19": torchvision.models.vgg19,
}


def create_vgg_readout(vgg, readout_type, input_size=None, num_classes=None):
    if readout_type == "dense":
        test_input = Variable(torch.zeros(1, 3, input_size, input_size))
        test_out = vgg.features(test_input)
        n_features = test_out.size(1) * test_out.size(2) * test_out.size(3)
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


class VGG(nn.Module):
    def __init__(
        self,
        input_size=64,
        pretrained=False,
        vgg_type="vgg19_bn",
        num_classes=200,
        readout_type="dense",
        input_channels=3,
    ):
        super(VGG, self).__init__()

        # load convolutional part of vgg
        assert vgg_type in VGG_TYPES, "Unknown vgg_type '{}'".format(vgg_type)
        vgg_loader = VGG_TYPES[vgg_type]
        vgg = vgg_loader(pretrained=pretrained)
        self.core = vgg.features
        self.input_channels = input_channels
        self.readout_type = readout_type

        # init fully connected part of vgg
        self.readout = create_vgg_readout(
            vgg, readout_type, input_size=input_size, num_classes=num_classes
        )
        self._init_readout_dense()

    def forward(self, x):
        if self.input_channels == 1:
            x = x.expand(-1, 3, -1, -1)
        core_out = self.core(x)
        if self.readout_type == "dense":
            core_out = core_out.view(core_out.size(0), -1)
        out = self.readout(core_out)
        return {"logits": out, "conv_rep": core_out}

    def freeze(self, selection=("core",)):
        if selection is True or "core" in selection:
            for param in self.core.parameters():
                param.requires_grad = False
        elif "readout" in selection:
            for param in self.readout.parameters():
                param.requires_grad = False

    def _init_readout_dense(self):
        for m in self.readout:
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
