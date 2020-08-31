# 判别器网络

import torch
import torch.nn as nn
import torch.nn.functional as F

class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 4, 2)
        self.conv2 = nn.Conv2d(32, 32, 4, 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 4, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 128, 4, 1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(128, 256, 4, 1)
        self.conv6 = nn.Conv2d(128, 3, 1, 1)
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.batch_norm(x) # BN 层
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = F.batch_norm(x) # BN 层
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv4(x)
        x = F.batch_norm(x) # BN 层
        x = F.relu(x)
        x = self.pool3(x)

        x = self.conv5(x)
        x = F.batch_norm(x) # BN 层
        x = F.relu(x)

        x = self.conv6(x)
        x = F.sigmoid(x)

DNet = D()
print(DNet)
