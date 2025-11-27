# ================================================
# 环境测试

# 输出当前 Python 解释器的绝对路径，确保使用的是正确的环境
import sys
print('System Version:', sys.version)
print(sys.executable)
print(sys.path)

# matplotlib ：常用的绘图库
import matplotlib
import matplotlib.pyplot as plt # For data viz
print(matplotlib.__version__)
print(matplotlib.__file__)

# PyTorch 及其相关库的版本信息和 CUDA 支持情况
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import torchaudio

print("nn_template 模块被导入")

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("Torchaudio version:", torchaudio.__version__)

# CUDA 运行时版本（对应 pytorch-cuda=12.4）
print("CUDA available:", torch.cuda.is_available())
print("CUDA version used by PyTorch:", torch.version.cuda)

# 当前 GPU 信息
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

# timm: PyTorch 图像模型库
import timm

# tqdm: 进度条库
# from tqdm.notebook import tqdm
from tqdm import tqdm

# numpy: 数值计算库
import numpy as np
print('Numpy version', np.__version__)

# pandas: 数据处理和分析库
import pandas as pd
print('Pandas version', pd.__version__)

# ================================================
# 设置随机种子，确保实验的可重复性
_ = torch.manual_seed(0)

# ================================================
# 自定义数据集类，继承自 PyTorch 的 Dataset 类
class PlayingCardDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        # ImageFolder 加载以特定文件夹结构组织的图像数据集
        self.data = ImageFolder(data_dir, transform=transform)

    # 数据集中有多少个示例
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

    # @property 是一个内置的装饰器 (decorator)，它的主要作用是将一个类的方法转换为一个“属性”，使得程序员可以像访问普通属性一样调用这个方法，而不需要使用括号 ()
    @property
    def classes(self):
        return self.data.classes

# ================================================
# 定义模型

class SimpleCardClassifer(nn.Module):
    def __init__(self, num_classes=53):
        super(SimpleCardClassifer, self).__init__()
        # Where we define all the parts of the model
        # pretrained=True 表示加载预训练权重，即使用在 ImageNet 上训练好的 EfficientNet-B0 模型
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)

        # 使用 预训练模型（比如 ResNet、VGG）进行迁移学习时的技巧：
        # nn.Sequential：把多个层（如卷积层、激活层、线性层、批归一化层等）按顺序串联起来，调用它时，会按照顺序(先将输入数据传入第一个参数(第一层)，然后将第一层输出传递给第二个参数)把输入数据传递给每一层，最后输出最后一层的结果
        # 当 * 用在函数调用时，表示把一个列表或元组的元素拆开，依次作为参数传入函数
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
            # self.base_model.children() 返回一个迭代器，包含模型的所有子模块（层）
            # list(...) 把迭代器转换为一个列表
            # list(self.base_model.children())[:-1] 获取除了最后一层(此处最后一层是 fc 层)之外的所有子模块，形成一个新的列表
            # *list(self.base_model.children())[:-1] 把这个新的列表拆开，作为多个参数传入 nn.Sequential()，从而创建一个新的顺序容器

        
        # 1280 是 EfficientNet-B0 模型在最后一个卷积层输出的特征图通道数
        enet_out_size = 1280
        # Make a classifier
        self.classifier = nn.Sequential(
            # nn.Flatten() 用于将多维张量展平成二维张量，形状从 [batch_size, 1280, 1, 1]（假设是卷积特征图）变成 [batch_size, 1280]
            nn.Flatten(),
            # 表示输入 enet_out_size 维的特征向量，输出 num_classes 维，代表 num_classes 个类别的预测分数
            nn.Linear(enet_out_size, num_classes)
        )
    
    def forward(self, x):
        # Connect these parts and return the output
        x = self.features(x)
        output = self.classifier(x)
        return output


# ================================================
# 数据准备

transform = transforms.Compose([
    # Resize((128, 128))：把输入图像缩放到 128 像素高、128 像素宽
    transforms.Resize((128, 128)),
    # ToTensor()：把图像从 PIL.Image 或 numpy.ndarray 格式转换为 PyTorch 张量（Tensor），并将图像的像素值从 0–255 缩放到 0–1 范围
    transforms.ToTensor(),
])

# transform = transforms.Compose([
#     # MINST 数据集的数值范围是 0-255 的灰度图像,经过 ToTensor() 转换后变为 0-1 范围的张量
#     transforms.ToTensor(),
#
#     # 归一化处理：使用均值 0.1307 和标准差 0.3081 对图像进行归一化
#     # MNIST 全体像素均值：0.1307
#     # MNIST 全体像素标准差：0.3081
#     transforms.Normalize((0.1307,), (0.3081,))
#     # MNIST 是灰度图像，因此均值和标准差都是单通道的元组
#     # 如果是 RGB 彩色图像，则需要提供三个通道的均值和标准差，例如：((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
#     # transforms.Normalize((mean_R, mean_G, mean_B), (std_R, std_G, std_B))
# ])

train_folder = '/home/cc/workspace/my_project/python_project/dataset/cards_dataset/train/'
valid_folder = '/home/cc/workspace/my_project/python_project/dataset/cards_dataset/valid/'
test_folder = '/home/cc/workspace/my_project/python_project/dataset/cards_dataset/test/'

train_dataset = PlayingCardDataset(train_folder, transform=transform)
val_dataset = PlayingCardDataset(valid_folder, transform=transform)
test_dataset = PlayingCardDataset(test_folder, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ================================================
# 模型实例化
model = SimpleCardClassifer(num_classes=53)

# ================================================
# ================================================
# 自定义 PyTorch 模型模板

# class MyModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         """
#         初始化模型
#         Args:
#             input_dim (int): 输入特征维度
#             hidden_dim (int): 隐藏层维度
#             output_dim (int): 输出类别数
#         """
#         super(MyModel, self).__init__()
        
#         # 定义网络层
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, output_dim)
        
#         # 可选：定义 Dropout
#         self.dropout = nn.Dropout(p=0.5)
    
#     def forward(self, x):
#         """
#         前向传播
#         Args:
#             x (torch.Tensor): 输入数据
#         Returns:
#             torch.Tensor: 输出结果
#         """
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = F.relu(self.fc2(x))
#         x = self.dropout(x)
#         x = self.fc3(x)
#         return x

# ================================================
# 自定义数据集模板

# transform = transforms.Compose([
#     # Resize((128, 128))：把输入图像缩放到 128 像素高、128 像素宽
#     transforms.Resize((128, 128)),
#     # ToTensor()：把图像从 PIL.Image 或 numpy.ndarray 格式转换为 PyTorch 张量（Tensor），并将图像的像素值从 0–255 缩放到 0–1 范围
#     transforms.ToTensor(),
# ])

# class MyDataset(Dataset):
#     def __init__(self, data, labels, transform=None):
#         """
#         初始化数据集
#         Args:
#             data (list, np.array, 或 tensor): 输入数据
#             labels (list, np.array, 或 tensor): 标签
#             transform (callable, optional): 对数据进行预处理
#         """
#         self.data = data
#         self.labels = labels
#         self.transform = transform

#     def __len__(self):
#         """
#         返回数据集的样本数量
#         """
#         return len(self.data)

#     def __getitem__(self, idx):
#         """
#         根据索引 idx 返回数据和标签
#         """
#         x = self.data[idx]
#         y = self.labels[idx]

#         if self.transform:
#             x = self.transform(x)

#         return x, y