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
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

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
# 训练循环

num_epochs = 5
train_losses, val_losses = [], []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 模型实例化
model = SimpleCardClassifer(num_classes=53)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc='Training loop'):

        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # loss 是一个标量张量（Tensor(1)），表示当前这个 batch 的平均损失值
        # .item() 把这个张量转换成 Python 数字（float）
        # labels.size(0) 是当前 batch 的大小（即有多少个样本）
        running_loss += loss.item() * labels.size(0)

    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)
    
    # 评估阶段
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation loop'):
            # Move inputs and labels to the device
            images, labels = images.to(device), labels.to(device)
         
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            
    val_loss = running_loss / len(val_loader.dataset)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss}, Validation loss: {val_loss}")

# ================================================
# 测试与可视化

import torch
import torchvision.transforms as transforms

# PIL：Python 图像处理库
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np

# 预处理图像
def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
        # .convert("RGB") 将图片转换为 RGB 格式（确保通道数为 3，即 [R,G,B]）
    return image, transform(image).unsqueeze(0)

# 预测函数
def predict(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.cpu().numpy().flatten()

# 可视化函数
def visualize_predictions(original_image, probabilities, class_names):
    fig, axarr = plt.subplots(1, 2, figsize=(14, 7))
    
    # Display image
    axarr[0].imshow(original_image)
    axarr[0].axis("off")
    
    # Display predictions
    axarr[1].barh(class_names, probabilities)
    axarr[1].set_xlabel("Probability")
    axarr[1].set_title("Class Predictions")
    axarr[1].set_xlim(0, 1)

    plt.tight_layout()
    plt.show()

# Example usage
test_image = "/home/cc/workspace/my_project/python_project/dataset/cards_dataset/test/five of diamonds/2.jpg"
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

original_image, image_tensor = preprocess_image(test_image, transform)
probabilities = predict(model, image_tensor, device)

# Assuming dataset.classes gives the class names
class_names = train_dataset.classes
visualize_predictions(original_image, probabilities, class_names)

# ================================================
# 批量测试与可视化

from glob import glob

# glob 用于 查找符合特定路径模式的文件
# glob 函数匹配路径 /home/cc/.cache/kagglehub/.../test/*/* 下的所有文件：
    # 第一个 * 对应 类别文件夹（如 cat/、dog/）
    # 第二个 * 对应 图片文件名（如 cat1.jpg、dog2.jpg）
# 返回值 test_images 是一个列表，每个元素都是图片的完整路径字符串
test_images = glob('/home/cc/workspace/my_project/python_project/dataset/cards_dataset/test/*/*')

# 返回值 test_examples 是一个长度为 10 的列表，每个元素是一个图片路径字符串
test_examples = np.random.choice(test_images, 10)

for example in test_examples:
    original_image, image_tensor = preprocess_image(example, transform)
    probabilities = predict(model, image_tensor, device)

    # Assuming dataset.classes gives the class names
    class_names = train_dataset.classes 
    visualize_predictions(original_image, probabilities, class_names)

# ================================================
# 评估模型在测试集上的性能

model.eval()
running_loss = 0.0
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc='Testing loop'):
        # Move inputs and labels to the device
        images, labels = images.to(device), labels.to(device)
     
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * labels.size(0)
test_loss = running_loss / len(test_loader.dataset)
print(f"Test loss: {test_loss}")

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