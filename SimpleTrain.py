'''
# 此程序为简易版完整训练程序

'''

# 读取 DIV2K 数据集
# 其中数据集又分为采用 bicubic 双三次采样和未知方式降采样缩小图像分辨率
# 且数据集已按照两种降低方式提前准备高清图 HR、2倍缩小图、3倍缩小图、4倍缩小图
# 故本次采用 HR 和 bicubic 降采样的 2(/3/4) 倍缩小图作为训练集和验证集

# 先采用 X2 倍缩小的图片进行训练


# ''''''''''''''''''''''''''''''cfg''''''''''''''''''''''''''''''
# 配置函数，包含各种训练参数的配置

BATCH_SIZE = 4
EPOCH_NUMBER = 5
TRAIN_ROOT = "/home/jichao/gitRes/Datasets/DIV2K/train"
TRAIN_LABEL = "/home/jichao/gitRes/Datasets/DIV2K/label"
VAL_ROOT = ""
VAL_LABEL = ""
TEST_ROOT = ""
TEST_LABEL = ""
crop_size_img = (768, 768)
crop_size_label = (1523, 1523)
# ''''''''''''''''''''''''''''''cfg''''''''''''''''''''''''''''''


# 1. 导入所需的包
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torchvision import transforms, datasets, utils
import torchvision.transforms.functional as ff
from datetime import datetime
from tensorboardX import SummaryWriter


# ''''''''''''''''''''''''''''''dataset.py''''''''''''''''''''''''''''''
# 数据预处理文件，重中之中，需要手敲一遍
# 1. 标签处理
# 2. 标签编码
# 3. 可视化编码过程
# 4. 定义预处理类

# 图片数据集处理
# return：img，label
class DIV2KDataset(data.Dataset):
    def __init__(self, file_path=[], crop_size_img=None, crop_size_label=None):
        """para:
            file_path(list): 数据和标签路径,列表元素第一个为图片路径，第二个为标签路径
        """
        # 1 正确读入图片和标签路径
        if len(file_path) != 2:
            raise ValueError("同时需要图片和标签文件夹的路径，图片路径在前")
        self.img_path = file_path[0]
        self.label_path = file_path[1]
        # 2 从路径中取出图片和标签数据的文件名保持到两个列表当中（程序中的数据来源）
        self.imgs = self.read_file(self.img_path)
        self.labels = self.read_file(self.label_path)
        # 3 初始化数据处理函数设置
        self.crop_size_img = crop_size_img
        self.crop_size_label = crop_size_label

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        # 从文件名中读取数据（图片和标签都是png格式的图像数据）
        img = Image.open(img)
        label = Image.open(label)

        img, label = self.center_crop(img, label, crop_size_img, crop_size_label)

        img, label = self.img_transform(img, label)
        # print('处理后的图片和标签大小：',img.shape, label.shape)
        sample = {'img': img, 'label': label}

        return sample

    def __len__(self):
        return len(self.imgs)

    def read_file(self, path):
        """从文件夹中读取数据"""
        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, img) for img in files_list]
        file_path_list.sort()
        return file_path_list

    def center_crop(self, data, label, crop_size_img, crop_size_label):
        """裁剪输入的图片和标签大小"""
        data = ff.center_crop(data, crop_size_img)
        label = ff.center_crop(label, crop_size_label)
        return data, label

    def img_transform(self, img, label):
        """对图片和标签做一些数值处理"""
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        img = transform(img)
        label = transform(label)

        return img, label
# ''''''''''''''''''''''''''''''dataset.py''''''''''''''''''''''''''''''


# 2. 构建网络结构
# 网络结构，将图像变大一倍
class AnNet(nn.Module):
    def __init__(self):
        super(AnNet, self).__init__()
        self.conv1 = nn.Conv2d(3,  64, 5, 1, 0)
        self.conv2 = nn.Conv2d(64, 32, 3, 1, 0)
        self.conv3 = nn.ConvTranspose2d(32, 3,  9, 2, 4)

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
AnNet = AnNet.to(device=device)

optimizer = optim.Adam(AnNet.parameters(), lr=0.001)
criterion = nn.MSELoss()

train_Data = DIV2KDataset([TRAIN_ROOT, TRAIN_LABEL], crop_size_img, crop_size_label)
train_Loader = data.DataLoader(train_Data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


# 4. 训练模型
starttime = datetime.now() # 开始计时

for epoch in range(EPOCH_NUMBER):
    print("开始第 %d 轮训练"%(epoch+1))

    train_loss = 0.0
    

    for i_batch, sample in enumerate(train_Loader):

        img = sample["img"].to(device)
        label = sample["label"].to(device)

        pred = AnNet(img)

        loss = criterion(pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        
        print('epoch{}|batch[{}/{}]|batch_loss {:.8f}|'.format(epoch+1, i_batch+1, len(train_Data)/BATCH_SIZE, loss.item()))
        
        # 使用 TensorboardX 记录损失图像
        with SummaryWriter() as w:
            w.add_scalar('scalar/test', loss.item(), (i_batch+1+200*epoch) )
            w.add_scalar('scalar/epoch', loss.item(), (i_batch+1+200*epoch) )

        if i_batch%100 == 99:
            # 保存原图结果
            img_Save_Name = "img_Epoch" + str(epoch+1) + "_Batch" + str(i_batch+1) + ".jpg"
            utils.save_image(img, img_Save_Name)
            # 保存预测结果
            pred_Save_Name = "pred_Epoch" + str(epoch+1) + "_Batch" + str(i_batch+1) + ".jpg"
            utils.save_image(pred, pred_Save_Name)
            # 保存标签结果
            label_Save_Name = "label_Epoch" + str(epoch+1) + "_Batch" + str(i_batch+1) + ".jpg"
            utils.save_image(label, label_Save_Name)

    endtime = datetime.now() # 结束计时
    # 打印运行时间
    print("RunTime: {}h-{}m-{}s".format(endtime.hour-starttime.hour, endtime.minute-starttime.minute, endtime.second-starttime.second))

# torch.save(AnNet, "AnNet.pth")
torch.save(AnNet.state_dict(), "Epoch" + str(EPOCH_NUMBER) + ".pth")












