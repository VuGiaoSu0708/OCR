import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18UNet(nn.Module):
    def __init__(self, weights=ResNet18_Weights.DEFAULT):
        super().__init__()
        backbone = torchvision.models.resnet18(weights=weights)

        # Extract different layers from ResNet
        self.encoder1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # 64
        self.pool = backbone.maxpool
        self.encoder2 = backbone.layer1  # 64
        self.encoder3 = backbone.layer2  # 128
        self.encoder4 = backbone.layer3  # 256
        self.encoder5 = backbone.layer4  # 512

        # Decoder with skip connections
        self.upconv5 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder5 = self._decoder_block(512, 256)  # Skip from encoder4

        self.upconv4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder4 = self._decoder_block(256, 128)  # Skip from encoder3

        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder3 = self._decoder_block(128, 64)   # Skip from encoder2

        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder2 = self._decoder_block(96, 32)    # Skip from encoder1

        self.upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)
        self.final_activation = nn.Sigmoid()

    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoding
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        # Decoding with skip connections
        d5 = self.upconv5(e5)
        diffY = e4.size()[2] - d5.size()[2]
        diffX = e4.size()[3] - d5.size()[3]
        d5 = F.pad(d5, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        d5 = torch.cat([d5, e4], dim=1)
        d5 = self.decoder5(d5)

        d4 = self.upconv4(d5)
        diffY = e3.size()[2] - d4.size()[2]
        diffX = e3.size()[3] - d4.size()[3]
        d4 = F.pad(d4, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        # Add padding for d3
        diffY = e2.size()[2] - d3.size()[2]
        diffX = e2.size()[3] - d3.size()[3]
        d3 = F.pad(d3, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        # Add padding for d2
        diffY = e1.size()[2] - d2.size()[2]
        diffX = e1.size()[3] - d2.size()[3]
        d2 = F.pad(d2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        out = self.final_conv(d1)
        return self.final_activation(out)
