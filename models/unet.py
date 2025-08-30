import torch
import torch.nn as nn

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )

class Up(nn.Module):
    """
    Up block: upsample from `in_ch` -> `out_ch` (via ConvTranspose2d),
    concat with encoder features (which have `out_ch` channels),
    then run conv_block(2*out_ch -> out_ch).
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # upsample in_ch -> out_ch
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = conv_block(2 * out_ch, out_ch)

    def forward(self, x, bridge):
        x = self.up(x)
        # if shape mismatch due to odd sizes, center crop bridge
        if x.size()[2:] != bridge.size()[2:]:
            diffY = bridge.size(2) - x.size(2)
            diffX = bridge.size(3) - x.size(3)
            bridge = bridge[:, :, diffY//2:bridge.size(2)-diffY+diffY//2, diffX//2:bridge.size(3)-diffX+diffX//2]
        x = torch.cat([bridge, x], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=6, base_c=32, out_channels=1):
        super().__init__()
        C = base_c
        # encoder
        self.enc1 = conv_block(in_channels, C)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(C, C*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(C*2, C*4)
        self.pool3 = nn.MaxPool2d(2)
        # bottleneck
        self.bottleneck = conv_block(C*4, C*8)
        # decoder (Up(in_ch_from_prev, out_ch_matching_encoder))
        self.up3 = Up(C*8, C*4)
        self.up2 = Up(C*4, C*2)
        self.up1 = Up(C*2, C)
        self.final = nn.Conv2d(C, out_channels, kernel_size=1)

    def forward(self, x):
        # encoder
        x1 = self.enc1(x)         # C
        p1 = self.pool1(x1)
        x2 = self.enc2(p1)        # 2C
        p2 = self.pool2(x2)
        x3 = self.enc3(p2)        # 4C
        p3 = self.pool3(x3)
        b  = self.bottleneck(p3)  # 8C
        # decoder
        u3 = self.up3(b, x3)      # out -> 4C
        u2 = self.up2(u3, x2)     # out -> 2C
        u1 = self.up1(u2, x1)     # out -> C
        out = self.final(u1)
        return out
