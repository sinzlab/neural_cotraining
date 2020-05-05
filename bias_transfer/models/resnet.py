from torch import nn
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.resnet import ResNet as DefaultResNet
import torch
import numpy as np

class ResNet(DefaultResNet):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        core_stride=1,
        conv_stem_kernel_size=3,
        conv_stem_padding=1,
        conv_stem_stride=1,
        max_pool_after_stem=False,
        advanced_init=False,
        adaptive_pooling=False,
    ):
        nn.Module.__init__(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3,
            self.inplanes,
            kernel_size=conv_stem_kernel_size,
            stride=conv_stem_stride,
            padding=conv_stem_padding,
            bias=False,
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        if max_pool_after_stem:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = nn.Identity()
        self.layer1 = self._make_layer(block, 64, layers[0], stride=core_stride)
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        if adaptive_pooling:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avgpool = nn.AvgPool2d(4)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        if advanced_init:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

            # Zero-initialize the last BN in each residual branch,
            # so that the residual branch starts with zeros, and each residual block behaves like an identity.
            # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
            if zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        nn.init.constant_(m.bn3.weight, 0)
                    elif isinstance(m, BasicBlock):
                        nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x


def resnet_builder(seed: int, config):
    # if config.self_attention:
    #     from .resnet_self_attention import ResNet, Bottleneck

    if "18" in config.type or "34" in config.type:
        assert not config.self_attention
        block = BasicBlock
    else:
        block = Bottleneck
    if "18" in config.type:
        num_blocks = [2, 2, 2, 2]
    elif  "26" in config.type:
        num_blocks = [1, 2, 4, 1]
    elif "34" in config.type:
        num_blocks = [3, 4, 6, 3]
    elif "38" in config.type:
        num_blocks = [2, 3, 5, 2]
    elif "50" in config.type:
        num_blocks = [3, 4, 6, 3]
    elif "101" in config.type:
        num_blocks = [3, 4, 23, 3]
    elif "152" in config.type:
        num_blocks = [3, 8, 36, 3]
    else:
        raise KeyError

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    model = ResNet(
        block,
        num_blocks,
        num_classes=config.num_classes,
        core_stride=config.core_stride,
        conv_stem_kernel_size=config.conv_stem_kernel_size,
        conv_stem_stride=config.conv_stem_stride,
        conv_stem_padding=config.conv_stem_padding,
        max_pool_after_stem=config.max_pool_after_stem,
        adaptive_pooling=config.adaptive_pooling,
        advanced_init=config.advanced_init,
        zero_init_residual=config.zero_init_residual
    )
    return model
