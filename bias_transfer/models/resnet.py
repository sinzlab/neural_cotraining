'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

Implementation from: https://github.com/kuangliu/pytorch-cifar
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_corr_matrix(x):
    x_flat = x.flatten(1,-1)
    centered = (x_flat - x_flat.mean()) / x_flat.std()
    out = (centered @ centered.transpose(0,1) ) / x_flat.size()[1]
    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            # TODO extract matrix from here as well
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, inputs):
        (x, compute_corr) = inputs
        corr_matrices = []
        out = self.conv1(x)
        if compute_corr:
            corr_matrices.append(compute_corr_matrix(out))
        out = F.relu(self.bn1())
        out = self.conv2(out)
        if compute_corr:
            corr_matrices.append(compute_corr_matrix(out))
        out = self.bn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out, corr_matrices


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            # TODO extract matrix from here as well
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, inputs):
        (x, compute_corr) = inputs
        corr_matrices = []
        out = self.conv1(x)
        if compute_corr:
            corr_matrices.append(compute_corr_matrix(out))
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        if compute_corr:
            corr_matrices.append(compute_corr_matrix(out))
        out = F.relu(self.bn2(out))
        out = self.conv3(out)
        if compute_corr:
            corr_matrices.append(compute_corr_matrix(out))
        out = self.bn3(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out, corr_matrices


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layers = nn.ModuleList()
        self.layers.append(self._make_layer(block, 64, num_blocks[0], stride=1))
        self.layers.append(self._make_layer(block, 128, num_blocks[1], stride=2))
        self.layers.append(self._make_layer(block, 256, num_blocks[2], stride=2))
        self.layers.append(self._make_layer(block, 512, num_blocks[3], stride=2))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, compute_corr: bool = False, seed: int = None):
        out = self.conv1(x)
        corr_matrices = []
        if compute_corr:
            corr_matrices.append(compute_corr_matrix(out))
        out = F.relu(self.bn1(out))
        for layer in self.layers:
            out, corr_matrices_l = layer((out, compute_corr))
            # passing arguments as tuple to get multiple args through Sequential
            corr_matrices += corr_matrices_l
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, corr_matrices

    def freeze(self, exclude_linear=True):
        for layer in [self.layers] + [self.conv1, self.bn1] \
                     + [self.linear] if not exclude_linear else []:
            for param in layer.parameters():
                param.requires_grad = False


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def resnet_builder(data_loader, seed=42, num_classes=10, *args, **kwargs):  # returns ResNet50 TODO: make variable
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
