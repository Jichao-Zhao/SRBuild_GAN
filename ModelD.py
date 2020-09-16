# 判别器网络

import torch
import torch.nn as nn
import torch.nn.functional as F
import CFG


class DNet(nn.Module):
    def __init__(self):
        super(DNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 4, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 4, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv6 = nn.Conv2d(256, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = torch.sigmoid(x)

        # 取均值
        meanX = torch.Tensor(CFG.BATCH_SIZE, 1)
        for i in range(CFG.BATCH_SIZE):
            meanX[i][0] = x[i].mean()
        return meanX
