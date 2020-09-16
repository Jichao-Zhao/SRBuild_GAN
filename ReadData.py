# 读取数据函数
# ''''''''''''''''''''''''''''''dataset.py''''''''''''''''''''''''''''''
# 数据预处理文件，重中之中，需要手敲一遍
# 1. 标签处理
# 2. 标签编码
# 3. 可视化编码过程
# 4. 定义预处理类

# 图片数据集处理
# return：img，label
class DIV2KDataset(Dataset):
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
