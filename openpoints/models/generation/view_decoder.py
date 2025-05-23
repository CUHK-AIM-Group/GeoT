import torch
import torch.nn as nn
import torch.nn.functional as F

from ..build import MODELS



@MODELS.register_module()
class ViewDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()


        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, 5, stride=4, padding=1, output_padding=1),
            nn.ReLU()
        )
        # self.layer1 = nn.Sequential(
        #     nn.ConvTranspose2d(in_channels, in_channels // 2, 3, stride=2, padding=1, output_padding=1),
        #     nn.ReLU()
        # )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels // 2, in_channels // 4, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels // 4, in_channels // 8, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels // 8, out_channels, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        

    def forward(self, feats_img):
        feats = self.layer1(feats_img)
        feats = self.layer2(feats)
        feats = self.layer3(feats)
        img = self.layer4(feats)
        return img
    

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        return self.sigmoid(x)

@MODELS.register_module()
class ViewDecoder_big(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()


        self.layer1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DoubleConv(in_channels, in_channels // 2, in_channels // 2)
        )
        self.layer2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DoubleConv(in_channels // 2, in_channels // 4, in_channels // 4)
        )
        self.layer3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DoubleConv(in_channels // 4, in_channels // 8, in_channels // 8)
        )
        self.layer4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DoubleConv(in_channels //8, in_channels // 8, in_channels // 8)
        )

        self.out_layer = OutConv(in_channels // 8, out_channels)
        

    def forward(self, feats_img):
        feats = self.layer1(feats_img)
        feats = self.layer2(feats)
        feats = self.layer3(feats)
        feats = self.layer4(feats)
        img = self.out_layer(feats)
        return img
    

@MODELS.register_module()
class ViewDecoder_ds(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()


        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels // 2, in_channels // 4, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels // 4, in_channels // 8, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels // 8, in_channels // 8, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )

        self.out1 = nn.Sequential(
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.out2 = nn.Sequential(
            nn.Conv2d(in_channels // 4, out_channels, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.out3 = nn.Sequential(
            nn.Conv2d(in_channels // 8, out_channels, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.out4 = nn.Sequential(
            nn.Conv2d(in_channels // 8, out_channels, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
        

    def forward(self, feats_img):
        imgs = []
        feats1 = self.layer1(feats_img)
        feats2 = self.layer2(feats1)
        feats3 = self.layer3(feats2)
        feats4 = self.layer4(feats3)

        img1 = self.out1(feats1)
        img2 = self.out2(feats2)
        img3 = self.out3(feats3)
        img4 = self.out4(feats4)

        imgs.append(img1)
        imgs.append(img2)
        imgs.append(img3)
        imgs.append(img4)

        return imgs