# 基于 GAN 网络的训练函数
# 文件顺序

# 1.导入必须的包
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from ModelD import DNet
from ModelG import GNet
import CFG
from ReadData import DIV2KDataset


# 2. 导入网络模型
device = ("cuda" if torch.cuda.is_available() else "cpu")
device = ("cpu")
DNet = DNet().to(device)
GNet = GNet().to(device)


# 3. 导入使用的数据集、网络结构、优化器、损失函数等
dataset = DIV2KDataset([CFG.TRAIN_DATA, CFG.TRAIN_LABEL], CFG.crop_size_img, CFG.crop_size_label)
data_loader = DataLoader(dataset)

# 损失函数
criterion = nn.BCELoss()
# 优化器
d_Optimizer = torch.optim.Adam(DNet.parameters(), lr=0.0003)
g_Optimizer = torch.optim.Adam(GNet.parameters(), lr=0.0003)


# 4. 训练模型
train_loss_g = 0.00
train_loss_d = 0.00
for epoch in range(CFG.EPOCH_NUMBER):
    for i_batch, sample in enumerate(data_loader):
        img = sample['img']
        label = sample['label']
        num_img = img.size(0)
        
        # 判别器训练流程
        # 真实数据和真实标签，虚假标签
        real_img = img.to(device)
        real_out = DNet(real_img)
        real_label = torch.ones(num_img, 1).to(device) # 标签不用读取到的高分辨图像
        d_loss_real = criterion(real_out, real_label)
        real_scores = real_out
        
        z = torch.randn(num_img, z_dimension).to(device)

        fake_img = G(z)
        fake_out = D(fake_img)
        fake_label = torch.zeros(num_img, 1).to(device)
        d_loss_fake = criterion(fake_out, fake_label)
        fake_scores = fake_out

        # 反向传播训练判别器
        d_loss = d_loss_real + d_loss_fake
        d_Optimizer.zero_grad()
        d_loss.backward()
        d_Optimizer.step()


        # 生成器训练流程
        z = torch.randn(num_img, z_dimension).to(device)

        fake_img = G(z)
        output = D(fake_img)
        g_loss = criterion(output, real_label)

        # 反向传播训练生成器
        g_Optimizer.zero_grad()
        g_loss.backward()
        g_Optimizer.setp()
        

        train_loss_d += d_loss.item()
        train_loss_g += g_loss.item()


        print("train_loss_d:", train_loss_d)
        print("train_loss_g:", train_loss_g)

# 5. 保存模型


# 6. 测试模型

# print(DNet)
# print(GNet)
