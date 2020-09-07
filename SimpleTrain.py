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
# 其中，原图为 (360,480)，裁剪为 (352,480)。因为 352 可以被之后的下采样整除。

BATCH_SIZE = 4
EPOCH_NUMBER = 2
TRAIN_ROOT = "/home/jichao/gitRes/Datasets/DIV2K/train"
TRAIN_LABEL = "/home/jichao/gitRes/Datasets/DIV2K/label"
VAL_ROOT = './CamVid/val'
VAL_LABEL = './CamVid/val_labels'
TEST_ROOT = './CamVid/test'
TEST_LABEL = './CamVid/test_labels'
class_dict_path = './CamVid/class_dict.csv'
crop_size = (768, 768)
# ''''''''''''''''''''''''''''''cfg''''''''''''''''''''''''''''''


# ''''''''''''''''''''''''''''''dataset.py''''''''''''''''''''''''''''''
# 数据预处理文件，重中之中，需要手敲一遍
# 1. 标签处理
# 2. 标签编码
# 3. 可视化编码过程
# 4. 定义预处理类

"""补充内容见 data process and load.ipynb"""
import os
import numpy as np
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms, datasets, utils
import torchvision.transforms.functional as ff


# 
# 图片数据集处理
# return：img，label
# 
class DIV2KDataset(data.Dataset):
    def __init__(self, file_path=[], crop_size=None):
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
        self.crop_size = crop_size

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        # 从文件名中读取数据（图片和标签都是png格式的图像数据）
        img = Image.open(img)
        label = Image.open(label)

        img, label = self.center_crop(img, label, self.crop_size)

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

    def center_crop(self, data, label, crop_size):
        """裁剪输入的图片和标签大小"""
        data = ff.center_crop(data, crop_size)
        label = ff.center_crop(label, (2284, 2284))
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


# 1. 导入所需的包
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# ImageFolder 
# from torch.utils import data
# from torchvision import transforms, datasets, utils

from datetime import datetime

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
        self.conv1 = nn.Conv2d(3,  64, 5, 1, 0)
        self.conv2 = nn.Conv2d(64, 32, 3, 1, 0)
        self.conv3 = nn.ConvTranspose2d(32, 3,  9, 3, 4)

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
batch_size = 4
num_epoch = 10

train_trans = transforms.Compose([
    transforms.CenterCrop(1536/2), # 中心位置切割
    transforms.ToTensor(),])
label_trans = transforms.Compose([
    transforms.CenterCrop(1536), # 中心位置切割
    transforms.ToTensor(),])

# 数据导入时不可随机，否则 train 和 label 将会不匹配
# train_data = datasets.ImageFolder("/home/jichao/gitRes/Datasets/DIV2K/train", transform=train_trans)
# train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=False)

# label_data = datasets.ImageFolder("/home/jichao/gitRes/Datasets/DIV2K/label", transform=label_trans)
# label_loader = data.DataLoader(label_data, batch_size=batch_size, shuffle=False)

train_Data = DIV2KDataset([TRAIN_ROOT, TRAIN_LABEL], crop_size)
train_Loader = data.DataLoader(train_Data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


# 4. 训练模型
for epoch in range(num_epoch):
    print("开始第 %d 轮训练"%(epoch+1))
    # AnNet.train(0)
    train_loss = 0.0

    for i_batch, sample in enumerate(train_Loader):
        print("开始第 %d 个 batch 训练"%(i_batch+1))
        starttime = datetime.now() # 开始计时

        img = sample["img"].to(device)
        label = sample["label"].to(device)

        pred = AnNet(img)

        loss = criterion(pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        endtime = datetime.now() # 结束计时

        # 保存原图结果
        img_Save_Name = "img" + str(i_batch+1) + ".jpg"
        utils.save_image(img, img_Save_Name)
        # 保存训练结果
        pred_Save_Name = "pred" + str(i_batch+1) + ".jpg"
        utils.save_image(pred, pred_Save_Name)
        # 打印运行时间
        print("RunTime: {}h-{}m-{}s-{}ms".format(endtime.hour-starttime.hour, endtime.minute-starttime.minute, endtime.second-starttime.second, (endtime.microsecond-starttime.microsecond)/1000))
        print('|batch[{}/{}]|batch_loss {:.8f}|'.format(i_batch+1, len(train_Data), loss.item()))












