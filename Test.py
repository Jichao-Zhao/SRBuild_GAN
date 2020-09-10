# 测试网络结构性能
# Set5, Set14

# 测试集文件夹文件路径
Set5_Path = "/home/jichao/gitRes/Datasets/Set5/"
Set14_Path  = "/home/jichao/gitRes/Datasets/Set14/"


import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import utils
from PIL import Image
import os
import torch.nn.functional as F


class ReadData(Dataset):
    def __init__(self, file_path):
        self.img_list = self.Read_File(file_path)
    
    def __getitem__(self, index):
        img_path = self.img_list[index]
        img = Image.open(img_path)
        img = self.img_transform(img)
        return img

    def __len__(self):
        return len(self.img_list)

    def Read_File(self, file_path):
        '''
        从文件夹中读取图片的路径，组合成列表并返回
        '''
        files_list = os.listdir(file_path)
        file_path_list = [os.path.join(file_path, img) for img in files_list]
        file_path_list.sort()
        return file_path_list

    def img_transform(self, img):
        transform_img = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(),
        ])
        return transform_img(img)


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


train_data = ReadData(Set5_Path)
train_loader = DataLoader(train_data, batch_size=1, shuffle=False)


AnNet = AnNet()
AnNet.load_state_dict(torch.load('Epoch1.pth'))

for i_batch, img in enumerate(train_loader):
    pred = AnNet(img)
    utils.save_image(pred, "imgPred"+ str(i_batch) +".png")






