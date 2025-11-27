from config import get_config, get_weights_file_path, latest_weights_file_path
from nn_template import model, train_loader, test_loader

import torch
import torch.nn as nn

from tqdm import tqdm

# 根据相对路径创建绝对路径的库
from pathlib import Path

def get_device():
    """
    检测并配置计算设备
    优先使用 CUDA (NVIDIA GPU)，其次是 MPS (Mac M1/M2/M3)，最后回退到 CPU
    Returns:
        torch.device: 配置好的 PyTorch 设备对象
    """
    # 1. 检测可用设备字符串
    if torch.cuda.is_available():
        device_type = "cuda"
    # 检查 MPS 是否可用 (Mac)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"
    
    print(f"Using device: {device_type}")

    # 2. 打印设备详细信息
    if device_type == 'cuda':
        # 注意：这里默认获取第 0 号 GPU 的信息
        # 原代码中的 device.index 在 device 仍为字符串时会报错，因此改为显式指定索引 0
        current_idx = 0
        print(f"Device name: {torch.cuda.get_device_name(current_idx)}")
        # 计算显存大小 (GB)
        mem_gb = torch.cuda.get_device_properties(current_idx).total_memory / 1024 ** 3
        print(f"Device memory: {mem_gb:.2f} GB")
    elif device_type == 'mps':
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")

    # 3. 实例化并返回设备对象
    return torch.device(device_type)

def train_model(config, train_loader, model):
    # Define the device
    device = get_device()

    # 确保权重文件夹存在
    # 在当前工作目录下创建一个名为 datasource_model_folder 的文件夹（如果不存在的话）
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)
    
    model.to(device)

    # 创建一个优化器对象 optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # 如果用户指定了一个模型在训练前预加载，则加载它
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    # latest_weights_file_path(config) 是一个函数，返回最新的权重文件路径或者返回 None(文件不存在才会返回 None)
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        # torch.load() 是 PyTorch 中用来加载序列化对象（通常是模型权重、训练状态等）的函数
        state = torch.load(model_filename, weights_only=False)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    # PyTorch 中定义交叉熵损失函数的类，计算预测的类别概率分布与真实标签之间的交叉熵损失
    # label_smoothing就是对样本的真实标签进行平滑处理，label_smoothing 是对 one-hot 标签（概率分布）进行平滑的技术
    # 平滑案例(label_smoothing = 0.1)：[0, 0, 1] -> [0.05, 0.05, 0.9]
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        # torch.cuda.empty_cache() 是 PyTorch 中的一个函数，用于清空当前 GPU 的缓存
        torch.cuda.empty_cache()
        # 将模型设置为训练模式，只影响模型中某些特定的层的行为，例如启用 dropout 和 batch normalization
        model.train()
        
        batch_iterator = tqdm(train_loader, desc=f"Processing Epoch {epoch:02d}")
        for images, labels in batch_iterator:

            images, labels = images.to(device), labels.to(device)

            # 将所有模型参数的 .grad 属性置为 None，从而清除梯度，为下一个 batch 做准备
            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)

            # Compute the loss using a simple cross entropy
            # .view(-1, vocab_size) 将张量展平为二维张量，-1 表示自动计算该维度的大小
            # 例如：如果 proj_output 的形状是 (B, seq_len, vocab_size)，那么 proj_output.view(-1, vocab_size) 的形状就是 (B * seq_len, vocab_size)
            loss = loss_fn(outputs, labels)
            # 用于在训练过程中更新进度条的后缀信息，以便实时显示当前的 loss 值
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Backpropagate the loss
            # 反向传播
            loss.backward()

            # Update the weights
            # 在执行了 loss.backward() 之后，所有模型参数的 .grad 属性中就已经存储了当前 batch 的梯度
            # optimizer.step() 用这些梯度来更新模型参数
            optimizer.step()

            global_step += 1

        # 在每个 epoch 结束时保存模型
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        # torch.save 序列化保存模型与训练状态到文件中
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

def test(test_loader, model: nn.Module):

    # Define the device
    device = get_device()

    correct = 0
    total = 0

    iterations = 0

    model.to(device)
    model.eval()

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images, labels = images.to(device), labels.to(device)

            # 使用其他模型时，可能需要调整这里的输入格式
            output = model(images)
            
            # enumerate 接受一个可迭代对象（列表、张量、生成器等）,返回 索引和元素 的对 (index, element)
            for idx, i in enumerate(output):
                # torch.argmax(i) → 返回张量 i 中最大值的索引
                if torch.argmax(i) == labels[idx]:
                    correct +=1
                total +=1
            iterations += 1
    accuracy = correct / total
    print(f'Accuracy: {round(accuracy * 100, 2)}%')

if __name__ == '__main__':
    config = get_config()
    train_model(config, train_loader, model)
    test(test_loader, model)