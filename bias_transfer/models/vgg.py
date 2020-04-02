import torch
from torch.autograd import Variable
import torchvision
import torch.nn as nn


VGG_TYPES = {'vgg11' : torchvision.models.vgg11,
             'vgg11_bn' : torchvision.models.vgg11_bn,
             'vgg13' : torchvision.models.vgg13,
             'vgg13_bn' : torchvision.models.vgg13_bn,
             'vgg16' : torchvision.models.vgg16,
             'vgg16_bn' : torchvision.models.vgg16_bn,
             'vgg19_bn' : torchvision.models.vgg19_bn,
             'vgg19' : torchvision.models.vgg19}

class Squeeze(nn.Module):
    def forward(self, input):
        return input.squeeze(-1)

class VGG(nn.Module):

    def __init__(self,
                 input_size=64,
                 pretrained=False,
                 vgg_type='vgg19_bn',
                 num_classes=200, classifier_type="conv"):
        super(VGG, self).__init__()

        # load convolutional part of vgg
        assert vgg_type in VGG_TYPES, "Unknown vgg_type '{}'".format(vgg_type)
        vgg_loader = VGG_TYPES[vgg_type]
        vgg = vgg_loader(pretrained=pretrained)
        self.core = vgg.features

        self.classifier_type = classifier_type

        # init fully connected part of vgg
        if classifier_type == "dense":
            test_input = Variable(torch.zeros(1,3,input_size,input_size))
            test_out = vgg.features(test_input)
            self.n_features = test_out.size(1) * test_out.size(2) * test_out.size(3)
            self.classifier = nn.Sequential(nn.Linear(self.n_features, 4096),
                                            nn.ReLU(True),
                                            nn.Dropout(),
                                            nn.Linear(4096, 4096),
                                            nn.ReLU(True),
                                            nn.Dropout(),
                                            nn.Linear(4096, num_classes)
                                           )
            self._init_classifier_dense()
        elif classifier_type == "conv":
            self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d((2, 2)),
                                        nn.Conv2d(512, 4096, 2, 2, bias=True),
                                        nn.ReLU(True),
                                        nn.Dropout(),
                                        nn.Conv2d(4096, 4096, 1, 1, bias=True),
                                        nn.ReLU(True),
                                        nn.Dropout(),
                                        nn.Conv2d(4096, num_classes, 1, 1, bias=True),
                                        Squeeze(), Squeeze())
            self._init_classifier_conv()

    def forward(self, x):
        x = self.core(x)
        if self.classifier_type == "dense":
            x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return {"logits" : x}

    def _init_classifier_dense(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


    def _init_classifier_conv(self):
        for m in self.classifier:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)