"""
Author: Coder729
Date: 2025/2/27
Description: Datasets
"""
from datasets import *

datasets = load_dataset("madao33/new-title-chinese")
# print(datasets) # 可以看到dataset中划分了训练集和测试集

# 多种划分方式
"""
dataset = load_dataset("madao33/new-title-chinese", split="train") # 只加载训练集
dataset = load_dataset("madao33/new-title-chinese", split="train[10:100]") # 加载训练集的前10到100个样本
dataset = load_dataset("madao33/new-title-chinese", split="train[:50%]") # 加载训练集的前50%个样本
dataset = load_dataset("madao33/new-title-chinese", split=["train[:50%]", "train[50%:]"]) # 加载训练集的前50%个样本和后50%个样本
"""

print(datasets["train"][0])