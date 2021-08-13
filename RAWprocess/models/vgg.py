"""
Encoder for few shot segmentation (VGG16)
"""

import torch
import torch.nn as nn

class Encoder_SegUnet(nn.Module):
    """
    Encoder for few shot segmentation

    Args:
        in_channels:
            number of input channels
        pretrained_path:
            path of the model for initialization
    """
    def __init__(self, in_channels, pretrained_path=None):
        super().__init__()
        self.pretrained_path = pretrained_path

        self.down_1 = conv_block_3d(in_channels, in_channels*2)
        self.pool_1 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.down_2 = conv_block_3d(in_channels*2, in_channels*4)
        self.pool_2 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.down_3 = conv_block_3d(in_channels*4, in_channels*8)
        self.pool_3 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)

        self.bridge = conv_block_3d(in_channels*8, in_channels*8)

        # self.trans_1 = conv_trans_block_3d(in_channels*16, in_channels*16)
        self.unpool_1 = nn.MaxUnpool3d(kernel_size=2, stride=2)
        self.gn_1 = nn.GroupNorm(in_channels*2, in_channels*8)
        self.up_1 = conv_block_3d(in_channels*16, in_channels*4)
        # self.trans_2 = conv_trans_block_3d(in_channels*8, in_channels*8)
        self.unpool_2 = nn.MaxUnpool3d(kernel_size=2, stride=2)
        self.gn_2 = nn.GroupNorm(in_channels*1, in_channels*4)
        self.up_2 = conv_block_3d(in_channels*8, in_channels*2)
        # self.trans_3 = conv_trans_block_3d(in_channels*4, in_channels*4)
        self.unpool_3 = nn.MaxUnpool3d(kernel_size=2, stride=2)
        self.gn_3 = nn.GroupNorm(in_channels, in_channels*2)
        self.up_3 = conv_block_3d(in_channels*4, in_channels*4)

        self.out = nn.Conv3d(in_channels*4, in_channels*8, kernel_size=3, stride=1, padding=1)




    def forward(self, x):
        down_1 = self.down_1(x) # -> [1, 2, 200, 160, 160]
        pool_1, idx1 = self.pool_1(down_1) # -> [1, 2, 100, 80, 80]
        
        down_2 = self.down_2(pool_1) # -> [1, 4, 100, 80, 80]
        pool_2, idx2 = self.pool_2(down_2) # -> [1, 4, 50, 40, 40]
        
        down_3 = self.down_3(pool_2) # -> [1, 8, 50, 40, 40]
        pool_3, idx3 = self.pool_3(down_3) # -> [1, 8, 25, 20, 20]
        
        # Bridge
        bridge = self.bridge(pool_3) # -> [1, 128, 4, 4, 4]
        
        # Up sampling
        unpool_1 = self.unpool_1(bridge, idx3) # -> [1, 8, 50, 40, 40]
        gn_1 = self.gn_1(unpool_1)
        concat_1 = torch.cat([gn_1, down_3], dim=1) # -> [1, 16, 50, 40, 40]
        up_1 = self.up_1(concat_1) # -> [1, 4, 50, 40, 40]
        
        unpool_2 = self.unpool_2(up_1, idx2) # -> [1, 4, 100, 80, 80]
        gn_2 = self.gn_2(unpool_2)
        concat_2 = torch.cat([gn_2, down_2], dim=1) # -> [1, 8, 100, 80, 80]
        up_2 = self.up_2(concat_2) # -> [1, 2, 100, 80, 80]
        
        unpool_3 = self.unpool_3(up_2, idx1) # -> [1, 2, 200, 160, 160]
        gn_3 = self.gn_3(unpool_3)
        concat_3 = torch.cat([gn_3, down_1], dim=1) # -> [1, 4, 200, 160, 160]
        up_3 = self.up_3(concat_3) # -> [1, 1, 200, 160, 160]

        out = self.out(up_3)
        return out

def conv_block_3d(in_dim, out_dim):
    if out_dim >= 4:
        return nn.Sequential(
            nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(int(out_dim / 4), out_dim),
            nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(int(out_dim / 4), out_dim),)
    else:
        return nn.Sequential(
            nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, out_dim),
            nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, out_dim),)

def conv_trans_block_3d(in_dim, out_dim):
    if out_dim >= 4:
        return nn.Sequential(
            nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(int(out_dim / 4), out_dim))
    else:
        return nn.Sequential(
            nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(1, out_dim))