import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, base_filters=32):
        super().__init__()
        self.down1 = DoubleConv(in_channels, base_filters)
        self.pool1 = nn.MaxPool2d(2)
        
        self.down2 = DoubleConv(base_filters, base_filters*2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.down3 = DoubleConv(base_filters*2, base_filters*4)
        self.pool3 = nn.MaxPool2d(2)
        
        self.down4 = DoubleConv(base_filters*4, base_filters*8)
        self.pool4 = nn.MaxPool2d(2)
        
        self.bottleneck = DoubleConv(base_filters*8, base_filters*16)
        
        self.up4 = nn.ConvTranspose2d(base_filters*16, base_filters*8, 2, stride=2)
        self.conv4 = DoubleConv(base_filters*16, base_filters*8)
        
        self.up3 = nn.ConvTranspose2d(base_filters*8, base_filters*4, 2, stride=2)
        self.conv3 = DoubleConv(base_filters*8, base_filters*4)
        
        self.up2 = nn.ConvTranspose2d(base_filters*4, base_filters*2, 2, stride=2)
        self.conv2 = DoubleConv(base_filters*4, base_filters*2)
        
        self.up1 = nn.ConvTranspose2d(base_filters*2, base_filters, 2, stride=2)
        self.conv1 = DoubleConv(base_filters*2, base_filters)
        
        self.out = nn.Conv2d(base_filters, out_channels, 1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
        
        d4 = self.down4(p3)
        p4 = self.pool4(d4)
        
        bn = self.bottleneck(p4)
        
        u4 = self.up4(bn)
        u4 = torch.cat([u4, d4], dim=1)
        c4 = self.conv4(u4)
        
        u3 = self.up3(c4)
        u3 = torch.cat([u3, d3], dim=1)
        c3 = self.conv3(u3)
        
        u2 = self.up2(c3)
        u2 = torch.cat([u2, d2], dim=1)
        c2 = self.conv2(u2)
        
        u1 = self.up1(c2)
        u1 = torch.cat([u1, d1], dim=1)
        c1 = self.conv1(u1)
        
        out = self.out(c1)
        return out

