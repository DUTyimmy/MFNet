import torch
import torch.nn as nn
import torch.nn.functional as f
from model.backbone.vgg_16 import VGG16


__all__ = ['MFNet']


class MFNet(nn.Module):
    def __init__(self):
        super(MFNet, self).__init__()

        # ------------------------  1st directive filter  ---------------------------- #
        self.conv1_1 = nn.Conv2d(512, 256, kernel_size=(3, 3), padding=(1, 1))
        self.conv1_3 = nn.Conv2d(256, 128, kernel_size=(3, 3), padding=(1, 1))
        self.conv1_5 = nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1))
        self.conv1_7 = nn.Conv2d(64, 32, kernel_size=(3, 3), padding=(1, 1))
        self.conv1_9 = nn.Conv2d(32, 1, kernel_size=(3, 3), padding=(1, 1))

        # ------------------------  2nd directive filter  ---------------------------- #
        self.conv2_1 = nn.Conv2d(512, 256, kernel_size=(3, 3), padding=(1, 1))
        self.conv2_3 = nn.Conv2d(256, 128, kernel_size=(3, 3), padding=(1, 1))
        self.conv2_5 = nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1))
        self.conv2_7 = nn.Conv2d(64, 32, kernel_size=(3, 3), padding=(1, 1))
        self.conv2_9 = nn.Conv2d(32, 1, kernel_size=(3, 3), padding=(1, 1))

        # ---------------------------  saliency decoder  ------------------------------ #
        self.side3_1_2 = nn.Conv2d(256, 128, kernel_size=(3, 3), padding=(1, 1))
        self.side3_2_2 = nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1))

        self.side4_1_2 = nn.Conv2d(512, 128, kernel_size=(3, 3), padding=(1, 1))
        self.side4_2_2 = nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1))

        self.side5_1_2 = nn.Conv2d(512, 128, kernel_size=(3, 3), padding=(1, 1))
        self.side5_2_2 = nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1))

        self.side3cat2 = nn.Conv2d(192, 64, kernel_size=(3, 3), padding=(1, 1))
        self.side3out2 = nn.Conv2d(64, 1, kernel_size=(3, 3), padding=(1, 1))

        # -----------------------------  others  -------------------------------- #
        self.sigmoid = nn.Sigmoid()
        self._initialize_weights()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

        # ---------------------------  shared encoder  ----------------------------- #
        self.vgg = VGG16()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # ---------------------------  shared encoder  ------------------------------ #
        x3, x4, x5 = self.vgg(x)

        # ------------------------  1st directive filter  ---------------------------- #
        sal1 = (self.conv1_9(self.up2(self.conv1_7(
                                 self.up2(self.conv1_5(
                                    self.up2(self.conv1_3(
                                        self.up2(self.conv1_1(x5))))))))))

        # ------------------------  2nd directive filter  ---------------------------- #
        sal2 = (self.conv2_9(self.up2(self.conv2_7(
                                self.up2(self.conv2_5(
                                    self.up2(self.conv2_3(
                                        self.up2(self.conv2_1(x5))))))))))

        # ---------------------------  saliency decoder  ------------------------------ #
        h_side3_2 = self.side3_2_2(self.side3_1_2(x3))
        h_side4_2 = self.side4_2_2(self.side4_1_2(x4))
        h_side5_2 = self.side5_2_2(self.side5_1_2(x5))
        # upsample to same size (1/4 original size)
        h_side5_2_up2 = self.upsample(h_side5_2)
        h_side5_2_up4 = self.upsample(h_side5_2_up2)
        h_side4_2_up2 = self.upsample(h_side4_2)

        # fusion
        side3_2 = self.side3out2(self.side3cat2(torch.cat((h_side5_2_up4, h_side4_2_up2, h_side3_2), 1)))
        # upsample to original size
        side3_2 = f.interpolate(side3_2, scale_factor=4, mode='bilinear', align_corners=False)

        return sal1.sigmoid(), sal2.sigmoid(), side3_2.sigmoid()
