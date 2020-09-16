# 生成器网络
# 包含两部分，深层网络和浅层网络

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter


class GNet(nn.Module):
    def __init__(self):
        super(GNet, self).__init__()

        # 4 倍上采样(Bicubic)
        self.UpSamp1 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bicubic', align_corners=True),
            nn.Conv2d(3, 64, 3, 1, 0))

        # 2 倍上采样(Bicubic)
        self.UpSamp2 =nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True),
            nn.Conv2d(3, 64, 3, 1, 0))

        # 深层网络，16 个残差块
        self.preTrain = nn.Conv2d(3, 64, 1, 1, 0)
        self.resBlock = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1))

        # 浅层网络，4 个卷积块
        self.shallowNet = nn.Sequential(
            nn.Conv2d(3, 64, 9, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 9, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 9, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 9, 1),
            nn.ReLU(inplace=True),
            )

        # 反卷积合并网络
        self.DeConv1 = nn.ConvTranspose2d(128, 64, 3, 1)
        self.DeConv2 = nn.ConvTranspose2d(128, 64, 3, 1)

        # 最后一层卷积
        self.Finally = nn.Conv2d(64, 3, 1, 1, 0)


    def forward(self, x):
        # 4 倍上采样
        x_4x = self.UpSamp1(x)
        # 2 倍上采样
        x_2x = self.UpSamp2(x)

        # 提取深层特征
        x_deep = self.preTrain(x)
        for i in range(16):
            x_deep += self.resBlock(x_deep)
        # x_deep0 = self.preTrain(x)
        # x_deep1 += self.resBlock(x_deep0) + x_deep0
        # x_deep2 += self.resBlock(x_deep1) + x_deep1
        # x_deep3 += self.resBlock(x_deep2) + x_deep2
        # x_deep += self.resBlock(x_deep3) + x_deep3

        # 浅层网络
        x_shallow = self.shallowNet(x)

        # 特征融合层
        x_DS = x_deep + x_shallow

        # 第一次反卷积
        x_Deconv1 = self.DeConv1(x_DS) + x_2x

        # 第二次反卷积
        x_Deconv2 = self.DeConv2(x_Deconv1) + x_4x

        # 最后一层卷积
        x = self.Finally(x_Deconv2)

        x = F.tanh(x)
        return x

