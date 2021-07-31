import torch.nn as nn
import torchvision


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.bn1_1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.bn1_2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.bn2_1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.bn2_2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.bn3_1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.bn3_2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.bn3_3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.bn4_1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.bn4_2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.bn4_3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.bn5_1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.bn5_2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.bn5_3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        conv1 = nn.Sequential()
        conv1.add_module('conv1_1', nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1)))
        conv1.add_module('bn1_1', self.bn1_1)
        conv1.add_module('relu1_1', nn.ReLU(inplace=True))
        conv1.add_module('conv1_2', nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)))
        conv1.add_module('bn1_2', self.bn1_2)
        conv1.add_module('relu1_2', nn.ReLU(inplace=True))
        self.conv1 = conv1
        conv2 = nn.Sequential()
        conv2.add_module('pool1', nn.AvgPool2d(2, stride=2))
        conv2.add_module('conv2_1', nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1)))
        conv2.add_module('bn2_1', self.bn2_1)
        conv2.add_module('relu2_1', nn.ReLU())
        conv2.add_module('conv2_2', nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1)))
        conv2.add_module('bn2_2', self.bn2_2)
        conv2.add_module('relu2_2', nn.ReLU())
        self.conv2 = conv2

        conv3 = nn.Sequential()
        conv3.add_module('pool2', nn.AvgPool2d(2, stride=2))
        conv3.add_module('conv3_1', nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1)))
        conv3.add_module('bn3_1', self.bn3_1)
        conv3.add_module('relu3_1', nn.ReLU())
        conv3.add_module('conv3_2', nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)))
        conv3.add_module('bn3_2', self.bn3_2)
        conv3.add_module('relu3_2', nn.ReLU())
        conv3.add_module('conv3_3', nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)))
        conv3.add_module('bn3_3', self.bn3_3)
        conv3.add_module('relu3_3', nn.ReLU())
        self.conv3 = conv3

        conv4 = nn.Sequential()
        conv4.add_module('pool3_1', nn.AvgPool2d(2, stride=2))
        conv4.add_module('conv4_1', nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1)))
        conv4.add_module('bn4_1', self.bn4_1)
        conv4.add_module('relu4_1', nn.ReLU())
        conv4.add_module('conv4_2', nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)))
        conv4.add_module('bn4_2', self.bn4_2)
        conv4.add_module('relu4_2', nn.ReLU())
        conv4.add_module('conv4_3', nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)))
        conv4.add_module('bn4_3', self.bn4_3)
        conv4.add_module('relu4_3', nn.ReLU())
        self.conv4 = conv4

        conv5 = nn.Sequential()
        conv5.add_module('pool4_1', nn.AvgPool2d(2, stride=2))
        conv5.add_module('conv5_1', nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)))
        conv5.add_module('bn5_1', self.bn5_1)
        conv5.add_module('relu5_1', nn.ReLU())
        conv5.add_module('conv5_2', nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)))
        conv5.add_module('bn5_2', self.bn5_2)
        conv5.add_module('relu5_2', nn.ReLU())
        conv5.add_module('conv5_3', nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)))
        conv5.add_module('bn5_3', self.bn5_3)
        conv5.add_module('relu5_3', nn.ReLU())
        self.conv5 = conv5

        vgg16_bn = torchvision.models.vgg16_bn(pretrained=True)
        self.copy_params_from_vgg16_bn(vgg16_bn)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # m.weight.data.zero_()
                nn.init.normal(m.weight.data, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        return x3, x4, x5

    def copy_params_from_vgg16_bn(self, vgg16_bn):
        features = [
            self.conv1.conv1_1, self.conv1.bn1_1, self.conv1.relu1_1,
            self.conv1.conv1_2, self.conv1.bn1_2, self.conv1.relu1_2,
            self.conv2.pool1,
            self.conv2.conv2_1, self.conv2.bn2_1, self.conv2.relu2_1,
            self.conv2.conv2_2, self.conv2.bn2_2, self.conv2.relu2_2,
            self.conv3.pool2,
            self.conv3.conv3_1, self.conv3.bn3_1, self.conv3.relu3_1,
            self.conv3.conv3_2, self.conv3.bn3_2, self.conv3.relu3_2,
            self.conv3.conv3_3, self.conv3.bn3_3, self.conv3.relu3_3,
            # self.conv3_4, self.bn3_4, self.relu3_4,
            self.conv4.pool3_1,
            self.conv4.conv4_1, self.conv4.bn4_1, self.conv4.relu4_1,
            self.conv4.conv4_2, self.conv4.bn4_2, self.conv4.relu4_2,
            self.conv4.conv4_3, self.conv4.bn4_3, self.conv4.relu4_3,
            # self.conv4_4, self.bn4_4, self.relu4_4,
            self.conv5.pool4_1,
            self.conv5.conv5_1, self.conv5.bn5_1, self.conv5.relu5_1,
            self.conv5.conv5_2, self.conv5.bn5_2, self.conv5.relu5_2,
            self.conv5.conv5_3, self.conv5.bn5_3, self.conv5.relu5_3,
        ]
        for l1, l2 in zip(vgg16_bn.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
            if isinstance(l1, nn.BatchNorm2d) and isinstance(l2, nn.BatchNorm2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
