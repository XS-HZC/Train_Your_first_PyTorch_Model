import kagglehub
import shutil
import os

# 使用 KaggleHub 下载数据集需要先进行身份验证

# 验证方法:在官网 https://www.kaggle.com/ 上注册账号
# 然后在 "Account" 页面生成传统 API Token
# 会下载一个 kaggle.json 文件，里面包含你的用户名和 API Key
# 将该文件放置在 ~/.kaggle/ 目录下 (Linux)

# 下载数据集
src_path = kagglehub.dataset_download("gpiosenka/cards-image-datasetclassification")

print("Path to dataset files:", src_path)

# 目标目录
dst_path = "/home/cc/workspace/my_project/python_project/dataset/cards_dataset"

os.makedirs(dst_path, exist_ok=True)

# 拷贝所有内容
shutil.copytree(src_path, dst_path, dirs_exist_ok=True)

print("Copied to:", dst_path)