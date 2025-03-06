"""
Author: Coder729
Date: 2025/3/6
Description: 
"""
import torch
import pandas as pd
from datasets import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments, pipeline

# 1. 加载数据集
datasets = Dataset.load_from_disk('./wiki_cn_filtered')
print(datasets[0])
datasets = datasets.select(range(300)) # 快速测试时使用

tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")

def preprocess_function(examples):
    return tokenizer(examples["completion"], max_length=512, truncation=True)

tokenized_datasets = datasets.map(preprocess_function, batched=True, remove_columns=datasets.column_names)
print(tokenized_datasets[0])

dataloader = DataLoader(tokenized_datasets, batch_size=2, collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)) #mlm=True表示使用Masked Language Modeling
print(next(iter(dataloader)))

model = AutoModelForMaskedLM.from_pretrained("hfl/chinese-macbert-base")

training_args = TrainingArguments(
    output_dir='./预训练模型',
    num_train_epochs=1,
    per_device_train_batch_size=16,
    logging_steps=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15),
    train_dataset=tokenized_datasets,
)

# trainer.train()

# 模型推理
# 模型推理
pipeline = pipeline('fill-mask', model='./预训练模型/checkpoint-19', tokenizer='hfl/chinese-macbert-base', device=0)

# 输入句子
sentence = "西安[MASK]通大[MASK]的博[MASK]馆馆长是锺[MASK]善"

# 获取推理结果
results = pipeline(sentence)

# 提取最大得分的词并填充[MASK]
filled_sentence = sentence
for i, result in enumerate(results):
    best_token = result[0]['token_str']  # 获取得分最高的词
    mask_index = filled_sentence.find("[MASK]")  # 找到第一个[MASK]
    filled_sentence = filled_sentence[:mask_index] + best_token + filled_sentence[mask_index + 6:]  # 填充[MASK]

print(filled_sentence)
