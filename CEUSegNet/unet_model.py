""" Full assembly of the parts to form the complete network """
import torch
import torch.nn.functional as F

from .unet_parts import *
import os


us_channels = [64, 64, 128, 256, 256]
ceus_channels = [64, 64, 128, 256, 256]


class UNet_USbranch(nn.Module):
    def __init__(self, n_channels):
        super(UNet_USbranch, self).__init__()
        self.n_channels = n_channels

        self.inc = DoubleConv(n_channels, us_channels[0])
        self.down1 = Down(us_channels[0], us_channels[1])
        self.down2 = Down(us_channels[1], us_channels[2])
        self.down3 = Down(us_channels[2], us_channels[3])
        self.down4 = Down(us_channels[3], us_channels[4])

        self.downsample2 = nn.Sequential(
            nn.Conv2d(us_channels[0], us_channels[1], (1, 1)),
            nn.BatchNorm2d(us_channels[1])
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(us_channels[1], us_channels[2], (1, 1)),
            nn.BatchNorm2d(us_channels[2])
        )
        self.downsample4 = nn.Sequential(
            nn.Conv2d(us_channels[2], us_channels[3], (1, 1)),
            nn.BatchNorm2d(us_channels[3])
        )
        self.downsample5 = nn.Sequential(
            nn.Conv2d(us_channels[3], us_channels[4], (1, 1)),
            nn.BatchNorm2d(us_channels[4])
        )

        self.out2 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2),
        )
        self.out3 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2),
        )
        self.out4 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2),
        )
        self.out5 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2),
        )


    def forward(self, x):
        x1 = self.inc(x)

        x2 = self.down1(x1)
        x2 = x2 + self.downsample2(x1)
        x2 = self.out2(x2)

        x3 = self.down2(x2)
        x3 = x3 + self.downsample3(x2)
        x3 = self.out3(x3)

        x4 = self.down3(x3)
        x4 = x4 + self.downsample4(x3)
        x4 = self.out4(x4)

        x5 = self.down4(x4)
        x5 = x5 + self.downsample5(x4)
        x5 = self.out5(x5)

        return x1, x2, x3, x4, x5


class UNet_dowm(nn.Module):
    def __init__(self, n_channels):
        super(UNet_dowm, self).__init__()
        self.n_channels = n_channels

        self.inc = DoubleConv(n_channels, ceus_channels[0])
        self.down1 = Down(ceus_channels[0], ceus_channels[1])
        self.down2 = Down(ceus_channels[1], ceus_channels[2])
        self.down3 = Down(ceus_channels[2], ceus_channels[3])
        self.down4 = Down(ceus_channels[3], ceus_channels[4])

        self.downsample2 = nn.Sequential(
            nn.Conv2d(us_channels[0], us_channels[1], (1, 1)),
            nn.BatchNorm2d(us_channels[1])
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(us_channels[1], us_channels[2], (1, 1)),
            nn.BatchNorm2d(us_channels[2])
        )
        self.downsample4 = nn.Sequential(
            nn.Conv2d(us_channels[2], us_channels[3], (1, 1)),
            nn.BatchNorm2d(us_channels[3])
        )
        self.downsample5 = nn.Sequential(
            nn.Conv2d(us_channels[3], us_channels[4], (1, 1)),
            nn.BatchNorm2d(us_channels[4])
        )

        self.out2 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2),
        )
        self.out3 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2),
        )
        self.out4 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2),
        )
        self.out5 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2),
        )

    def forward(self, x):
        x1 = self.inc(x)

        x2 = self.down1(x1)
        x2 = x2 + self.downsample2(x1)
        x2 = self.out2(x2)

        x3 = self.down2(x2)
        x3 = x3 + self.downsample3(x2)
        x3 = self.out3(x3)

        x4 = self.down3(x3)
        x4 = x4 + self.downsample4(x3)
        x4 = self.out4(x4)

        x5 = self.down4(x4)
        x5 = x5 + self.downsample5(x4)
        x5 = self.out5(x5)

        return x1, x2, x3, x4, x5


class UNet_up(nn.Module):
    def __init__(self, n_classes):
        super(UNet_up, self).__init__()
        self.n_classes = n_classes

        self.up1 = Up(256 + 256, 128)
        self.up2 = Up(128 + 128, 64)
        self.up3 = Up(64 + 64, 64)
        self.up4 = Up(64 + 64, 64)
        self.outc = OutConv(64, n_classes)
        self.act = nn.Sigmoid()

    def forward(self, x1, x2, x3, x4, x5):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        y = self.act(logits)
        return y


class modalityChannelAttentionModuleOriginstyle(nn.Module):
    def __init__(self, ceus_channels, us_channels):
        super(modalityChannelAttentionModuleOriginstyle, self).__init__()
        self.ceus_channels = ceus_channels
        self.us_channels = us_channels
        self.factor = torch.sqrt(torch.tensor(1 / (ceus_channels + us_channels) // 4))
        self.GAP_ceus = nn.AdaptiveAvgPool2d((1, 1))
        self.GAP_us = nn.AdaptiveAvgPool2d((1, 1))
        self.w_ceus = nn.Linear(in_features=ceus_channels, out_features=ceus_channels, bias=False)
        self.w_us = nn.Linear(in_features=us_channels, out_features=ceus_channels, bias=False)
        self.softmax = nn.Softmax(-1)

    def forward(self, x_ceus, x_us):
        v_ceus = self.GAP_ceus(x_ceus).squeeze(-1).squeeze(-1)
        v_us = self.GAP_us(x_us).squeeze(-1).squeeze(-1)

        v_ceus = self.w_ceus(v_ceus)
        v_us = self.w_us(v_us)

        weight = self.softmax(torch.mul(v_ceus, v_us) * self.factor).unsqueeze(-1).unsqueeze(-1)
        y_ceus = weight * x_ceus + x_ceus
        y_us = x_us

        return y_ceus, y_us
        # return x_ceus, x_us


class multiScaleFusionBlock(nn.Module):
    def __init__(self, channels):
        super(multiScaleFusionBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels // 4, kernel_size=(3, 3), dilation=(1, 1), padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels // 4, kernel_size=(3, 3), dilation=(2, 2), padding=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels // 4, kernel_size=(3, 3), dilation=(4, 4), padding=4),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels // 4, kernel_size=(3, 3), dilation=(8, 8), padding=8),
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)

        conv = torch.cat([conv1, conv2, conv3, conv4], dim=1)
        return conv



class branchFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(branchFusionModule, self).__init__()
        self.fusion_early = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, padding=0, kernel_size=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.multiScaleFusion = multiScaleFusionBlock(out_channels)

        self.fusion_late = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, padding=0, kernel_size=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x1_ceus, x1_us):
        x = torch.cat([x1_ceus, x1_us], dim=1)
        fusion = self.fusion_early(x)
        out = self.multiScaleFusion(fusion)
        y = self.fusion_late(out)
        
        return y


class CEUSegNet(nn.Module):
    def __init__(self):
        super(CEUSegNet, self).__init__()
        self.US_branch = UNet_USbranch(n_channels=1) # gray US image
        self.CEUS_dowm_branch = UNet_dowm(n_channels=1)
        self.CEUS_up_branch = UNet_up(1)

        self.fusion5 = branchFusionModule(256 + 256, 256) # CFF
        self.fusion4 = branchFusionModule(256 + 256, 256)
        self.fusion3 = branchFusionModule(128 + 128, 128)
        self.fusion2 = branchFusionModule(64 + 64, 64)
        self.fusion1 = branchFusionModule(64 + 64, 64)

        self.att1 = modalityChannelAttentionModuleOriginstyle(64, 64) # CSA
        self.att2 = modalityChannelAttentionModuleOriginstyle(64, 64)
        self.att3 = modalityChannelAttentionModuleOriginstyle(128, 128)
        self.att4 = modalityChannelAttentionModuleOriginstyle(256, 256)
        self.att5 = modalityChannelAttentionModuleOriginstyle(256, 256)


    def forward(self, x):
        # B * 3 * C * W * H
        x_us = x[:, 0, :, :, :]
        x_ceus = x[:, 1, :, :, :]

        x1_ceus, x2_ceus, x3_ceus, x4_ceus, x5_ceus = self.CEUS_dowm_branch(x_ceus)
        x1_us, x2_us, x3_us, x4_us, x5_us = self.US_branch(x_us)  # B * C * H * W

        x1_ceus, x1_us = self.att1(x1_ceus, x1_us)
        x2_ceus, x2_us = self.att2(x2_ceus, x2_us)
        x3_ceus, x3_us = self.att3(x3_ceus, x3_us)
        x4_ceus, x4_us = self.att4(x4_ceus, x4_us)
        x5_ceus, x5_us = self.att5(x5_ceus, x5_us)


        x1, x2, x3, x4, x5 = self.fusion1(x1_ceus, x1_us), self.fusion2(x2_ceus, x2_us), self.fusion3(x3_ceus, x3_us), self.fusion4(x4_ceus, x4_us), self.fusion5(x5_ceus, x5_us)

        o = self.CEUS_up_branch(x1, x2, x3, x4, x5) # B * C * H * W

        return o  # B * 1 * H * W


