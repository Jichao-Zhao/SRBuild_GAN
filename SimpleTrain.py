'''
# 此程序为简易版完整训练程序

'''

# 读取 DIV2K 数据集
# 其中数据集又分为采用 bicubic 双三次采样和未知方式降采样缩小图像分辨率
# 且数据集已按照两种降低方式提前准备高清图 HR、2倍缩小图、3倍缩小图、4倍缩小图
# 故本次采用 HR 和 bicubic 降采样的 2(/3/4) 倍缩小图作为训练集和验证集

# 先采用 X2 倍缩小的图片进行训练


# 1. 导入所需的包
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# ImageFolder 
from torch.utils import data
from torchvision import transforms, datasets, utils

'''
# 路径记录
# 此路径为 X2 倍训练集路径
Path_Train_LR_bicubic = "/home/jichao/gitRes/Datasets/DIV2K/train"
Path_Train_HR = "/home/jichao/gitRes/Datasets/DIV2K/label"

'''


# 2. 构建网络结构
# 网络结构，将图像变大一倍
class AnNet(nn.Module):
    def __init__(self):
        super(AnNet, self).__init__()
        self.conv1 = nn.Conv2d(3,  64, 9, 1, 4)
        self.conv2 = nn.Conv2d(64, 64, 1, 1, 0)
        self.conv3 = nn.Conv2d(64, 3,  5, 1, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        return x


# 3. 读取数据
AnNet = AnNet()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# AnNet = AnNet.to(device=device)

optimizer = optim.Adam(AnNet.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
batch_size = 8
num_epoch = 10

train_trans = transforms.Compose([
    transforms.CenterCrop(1536/2), # 中心位置切割
    transforms.ToTensor(),])
label_trans = transforms.Compose([
    transforms.CenterCrop(1536), # 中心位置切割
    transforms.ToTensor(),])

# 数据导入时不可随机，否则 train 和 label 将会不匹配
train_data = datasets.ImageFolder("/home/jichao/gitRes/Datasets/DIV2K/train", transform=train_trans)
train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=False)

label_data = datasets.ImageFolder("/home/jichao/gitRes/Datasets/DIV2K/label", transform=label_trans)
label_loader = data.DataLoader(label_data, batch_size=batch_size, shuffle=False)


# 4. 训练模型
for epoch in range(num_epoch):
    AnNet.train(0)
    train_loss = 0.0

    # train_loader = iter(train_loader)

    for i_batch, (img, _) in enumerate(train_loader):
        print("开始第 %d 轮训练"%(epoch+1))
        optimizer.zero_grad()
        # img = img.to(device)

        pred = AnNet(img)

        # 保存训练结果
        pred_Save_Name = "pred" + str(i_batch+1) + ".jpg"
        utils.save_image(pred, pred_Save_Name)

        for j_batch, (label, _) in enumerate(label_loader):
            


        # loss = criterion(pred, label)
        optimizer.step()


        # print(pred)
        # print(type(img))

for i_batch, img in enumerate(label_loader):
    if i_batch == 0:
        utils.save_image(img[0], 'label.png')
    break




