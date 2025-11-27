from pathlib import Path

def get_config():
    return {
        # 批次大小
        "batch_size": 8,
        # 训练轮数
        "num_epochs": 5,
        # 学习率
        "lr": 10**-4,
        
        # datasource 指定数据源名称
        "datasource": 'data_name',

        # 模型保存到名为 weights 的文件夹中
        "model_folder": "weights",
        # 模型保存的文件名
        "model_basename": "tmodel_",
        # 重新启用训练时，加载的模型文件名，例如训练时程序崩溃了，重新运行时加载的模型文件名
        "preload": "latest",
    }

# 获取模型文件路径
def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# 寻找 weights 文件夹中最新的权重文件
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])