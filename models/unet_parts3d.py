# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv3d(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv3d, self).__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv3d(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv3d = double_conv3d(in_ch, out_ch)

    def forward(self, x):
        x = self.conv3d(x)
        return x


class down3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down3d, self).__init__()
        self.mpconv3d = nn.Sequential(
            nn.MaxPool3d(2),
            double_conv3d(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv3d(x)
        return x


class up3d(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up3d, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up3d =  nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up3d = nn.ConvTranspose3d(in_ch , in_ch , 2, stride=2)

        self.conv3d = double_conv3d(in_ch, out_ch)

    def forward(self, x):
        x = self.up3d(x)
        x = self.conv3d(x)

        # input is CHW

        return x


class outconv3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv3d, self).__init__()
        self.conv3d = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv3d(x)
        return x
