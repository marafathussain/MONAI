import logging
import math
import torch
import torch.nn.functional as F
from torch import nn
from meta_modules import *

class UNet(MetaModule):
    def __init__(self, in_channels=1, out_channels=1, depth=5, pretrained=False, wf=4,
                 padding=True, batch_norm=False, up_mode='upsample'):
        
        super(UNet, self).__init__()
        
        self.padding        = padding
        self.depth          = depth
        self.down_path      = nn.ModuleList()
        self.out_channels   = out_channels
        
        self.down1 = UNetConvBlock(1    ,   16, padding, batch_norm)
        self.down2 = UNetConvBlock(16   ,  32, padding, batch_norm)
        self.down3 = UNetConvBlock(32  ,  64, padding, batch_norm)
        self.down4 = UNetConvBlock(64  ,  128, padding, batch_norm)
        self.down5 = UNetConvBlock(128  , 256, padding, batch_norm)
        
        self.decoder = UNetDecoder(depth, 256, up_mode, padding, batch_norm, wf, out_channels)
        
        if not pretrained:
            self.reset_parameters(self.down1, bn_gama=1.)
            self.reset_parameters(self.down2, bn_gama=1.)
            self.reset_parameters(self.down3, bn_gama=1.)
            self.reset_parameters(self.down4, bn_gama=1.)
            self.reset_parameters(self.down5, bn_gama=1.)
            
    def reset_parameters(self, module, bn_gama=0.):
        for m in module.modules():
            if isinstance(m, MetaConv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, bn_gama)
                nn.init.constant_(m.bias, 0.)
                
    def forward(self, x):
        blocks = []
        
        x = self.down1(x)           #16x16x96x96
        blocks.append(x)
        x = F.max_pool3d(x, 2)      #16x16x48x48
        
        x = self.down2(x)           #16x32x48x48
        blocks.append(x)
        x = F.max_pool3d(x, 2)      #16x32x24x24
        
        x = self.down3(x)           #16x64x24x24
        blocks.append(x)
        x = F.max_pool3d(x, 2)      #16x64x12x12
        
        x = self.down4(x)           #16x128x12x12
        blocks.append(x)
        x = F.max_pool3d(x, 2)      #16x128x6x6
        
        x = self.down5(x)           #16x256x6x6
        
        out = self.decoder(x, blocks)
        return out
    
class UNetDecoder(MetaModule):
    def __init__(self, depth, prev_channels, up_mode, padding, batch_norm, wf, n_classes):
        super(UNetDecoder, self).__init__()
        
        self.up_path = nn.ModuleList()
        for i in reversed(range(depth-1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i),
                            up_mode, padding, batch_norm))
            
            prev_channels = 2 ** (wf + i)
            
        self.last = MetaConv3d(prev_channels, n_classes, kernel_size=1)
        
    def forward(self, x, blocks):
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])
            
        out = self.last(x)
        return out

    
class UNetConvBlock(MetaModule):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []
        
        #...conv1
        block.append(MetaConv3d(in_size, out_size,kernel_size=3, padding=int(padding)))
        block.append(nn.LeakyReLU())
        if batch_norm: block.append(nn.BatchNorm3d(out_size))
        
        #...conv2
        block.append(MetaConv3d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.LeakyReLU())
        if batch_norm: block.append(nn.BatchNorm3d(out_size))
        
        self.block = nn.Sequential(*block)
        
    def forward(self, x):
        out = self.block(x)
        return out

    
class UNetUpBlock(MetaModule):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = MetaConvTranspose3d(in_size, out_size, kernel_size=2, stride=2)
            
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='trilinear', scale_factor=2,align_corners=False),
                MetaConv3d(in_size, out_size, kernel_size=1),
            )
            
        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)
        
    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width,_ = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1])
        ]
    
    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)
        
        return out
    
