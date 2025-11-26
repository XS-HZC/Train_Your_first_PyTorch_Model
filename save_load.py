import torch
import torch.nn as nn

# torch.save(arg, PATH)
# arg 可以是张量、模型或任何字典
# torch.load(PATH)
# 
# 使用案例：只保存模型参数（state_dict）
# torch.save(model.state_dict(), PATH)
# model2 = TheModelClass(*args, **kwargs)
# model2.load_state_dict(torch.load(PATH))
# model2.eval()  # 如果加载的是模型参数，需要调用 eval() 方法将模型设置为评估模式

# 模型案例代码
class Model(nn.Module):
    def __init__(self, n_input_features=10):
        super(Model, self).__init__()
        self.fc = nn.Linear(n_input_features, 1)  # 假设输入特征为10，输出特征为2

    def forward(self, x):
        y_pred = torch.sigmoid(self.fc(x))  # 使用sigmoid激活函数
        return y_pred

model = Model(n_input_features=6)

# print(model.state_dict())  # 打印模型参数字典

# for param in model.parameters():
#     print(param)

FILE = "model.pth"
torch.save(model.state_dict(), FILE)  # 只保存模型参数

model2 = Model(n_input_features=6)

# for param in model2.parameters():
#     print(param)

model2.load_state_dict(torch.load(FILE))  # 加载模型参数
model2.eval()  # 设置为评估模式

# for param in model2.parameters():
#     print(param)

# ========================================================================
# 检查点的保存和加载
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

optimizer2 = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0)

print()
print("optimizer:", optimizer.state_dict())  # 打印优化器状态字典
print()
print("optimizer2:", optimizer2.state_dict())  # 打印优化器状态字典
print()
print("model:", model.state_dict())  # 打印模型参数字典
print()
print("model2:", model2.state_dict())  # 打印模型参数字典
print()

checkpoint = {
    'epoch': 10,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}

torch.save(checkpoint, "checkpoint.pth")

loaded_checkpoint = torch.load("checkpoint.pth")
model2.load_state_dict(loaded_checkpoint['model_state_dict'])
optimizer2.load_state_dict(loaded_checkpoint['optimizer_state_dict'])

print("==========================")
print()

print("optimizer:", optimizer.state_dict())  # 打印优化器状态字典
print()
print("optimizer2:", optimizer2.state_dict())  # 打印优化器状态字典
print()
print("model:", model.state_dict())  # 打印模型参数字典
print()
print("model2:", model2.state_dict())  # 打印模型参数字典
print()

# ========================================================================
# GPU 上保存模型并在 CPU 上加载的情况
device = torch.device("cuda")
model.to(device)  # 将模型移动到 GPU
torch.save(model.state_dict(), "model_gpu.pth")  # 保存模型参数到 GPU

device = torch.device("cpu")
model2 = Model(n_input_features=6)  # 创建一个新的模型实例
model2.load_state_dict(torch.load("model_gpu.pth"), map_location=device)  # 从 GPU 加载模型参数到 CPU

# ========================================================================
# GPU 上保存模型并在 GPU 上加载的情况
device = torch.device("cuda")
model.to(device)  # 将模型移动到 GPU
torch.save(model.state_dict(), "model_gpu.pth")  # 保存模型参数到 GPU

model2 = Model(n_input_features=6)  # 创建一个新的模型实例
model2.load_state_dict(torch.load("model_gpu.pth"))
model2.to(device)  # 将模型移动到 GPU

# ========================================================================
# CPU 上保存模型并在 GPU 上加载的情况
torch.save(model.state_dict(), "model_cpu.pth")  # 保存模型参数到 CPU
device = torch.device("cuda")
model2 = Model(n_input_features=6)  # 创建一个新的模型实例
model2.load_state_dict(torch.load("model_cpu.pth", map_location=device))  # 从 CPU 加载模型参数到 GPU
model2.to(device)  # 将模型移动到 GPU