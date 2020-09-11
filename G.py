# 生成器网络
# 包含两部分，深层网络和浅层网络

import torch
import torch.nn as nn
import torch.nn.functional as F

class GNet(nn.Module):
    def __init__(self):
        super(GNet, self).__init__()
        self.shallowNet = nn.Sequential(
            nn.Conv2d(3, 64, 9, 1),
            nn.Conv2d(64, 64, 9, 1),
            nn.Conv2d(64, 64, 9, 1),
            nn.Conv2d(64, 64, 9, 1)
        )
        self.resBlock = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1),
        )
        self.DeConv1 = nn.ConvTranspose2d(128, 64, 3, 1)
        self.DeConv2 = nn.ConvTranspose2d(128, 64, 3, 1)

        self.UpSamp1 = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        self.UpSamp2 = nn.Upsample(scale_factor=4, mode='bicubic', align_corners=True)


    def forward(self, x):
        x_shallow = self.shallowNet(x) # 提取浅层特征

        # 提取深层特征
        x_deep = self.resBlock(x)
        for i in range(16):
            x_deep = self.resBlock(x_deep)
        # 特征进行融合
        x = torch.cat(x_shallow, x_deep)

        # 第一次反卷积
        x = self.DeConv1(x)
        # 2 倍上采样层 Bicubic
        x = self.UpSamp1(x)

        # 第二次反卷积
        x = self.DeConv2(x)
        # 4 倍上采样层 Bicubic
        x = self.UpSamp2(x)




