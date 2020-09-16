# 基于 GAN 网络的训练函数


# 1.导入必须的包
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import D
import G


# 2. 搭建网络模型
device = ("cuda" if torch.cuda.is_available() else "cpu")
DNet = D.DNet().to(device)
GNet = G.GNet().to(device)


# 3. 导入使用的数据集、网络结构、优化器、损失函数等
dataset = 
data_loader = DataLoader(dataset)

# 损失函数
criterion = nn.BCELoss()
# 优化器
d_Optimizer = torch.optim.Adam(DNet.parameters(), lr=0.0003)
g_Optimizer = torch.optim.Adam(GNet.parameters(), lr=0.0003)


# 4. 训练模型
for epoch in range(NUM_EPOCH):
    for i_batch, (img, _) = enumerate(data_loader):
        # 训练判别器
        # 真实数据
        real_img = img.to(device)
        real_label = 
        # 虚假数据

        # 训练生成器
        



# 5. 保存模型


# 6. 测试模型

print(DNet)
print(GNet)
