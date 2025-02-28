"""
Author: Coder729
Date: 2025/2/27
Description: Evaluate
"""
import sys
import os
import evaluate
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split

accuracy = evaluate.load("accuracy")
print(accuracy.description)

result = accuracy.compute(references=[1, 0, 1, 1, 0], predictions=[1, 1, 0, 1, 0])
print(result)

# 联合评价指标
classification_matrix = evaluate.combine(["accuracy", "f1", "precision", "recall"])
result_matrix = classification_matrix.compute(references=[1, 0, 1, 1, 0], predictions=[1, 1, 0, 1, 0])
print(result_matrix) # 可以看到，结果中包含了accuracy、f1、precision、recall四个指标的结果

# 实战：使用evaluate模块修改04_Model/实战.py中的手动评估

class ClassificationDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.data = pd.read_csv('../04_Model/datasets/ChnSentiCorp_htl_all.csv')
        self.data = self.data.dropna()  # 去掉空值
    def __getitem__(self, index):
        return self.data.iloc[index]["review"], self.data.iloc[index]["label"]
    def __len__(self):
        return len(self.data)

tokenizer = AutoTokenizer.from_pretrained("../04_Model/rbt3")

# 划分数据集
trainset, valset = random_split(ClassificationDataset(), lengths=[0.8, 0.2])
print(len(trainset), len(valset))

def collate_fn(batch):
    texts, labels = [], []
    for item in batch:
        texts.append(item[0])
        labels.append(item[1])
    inputs = tokenizer(texts, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    inputs["labels"] = torch.tensor(labels)
    return inputs

trainset_loader = DataLoader(trainset, batch_size=32, shuffle=True, collate_fn=collate_fn)
valset_loader = DataLoader(valset, batch_size=32, shuffle=False, collate_fn=collate_fn)

model = AutoModelForSequenceClassification.from_pretrained("../04_Model/rbt3").to('cuda')
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
classification_matrix_实战 = evaluate.combine(["accuracy", "f1"])

def evaluate(loader):
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to('cuda') for k, v in batch.items()}
            outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)
            classification_matrix_实战.add_batch(predictions=predictions.long(), references=batch["labels"].long())
    return classification_matrix_实战.compute()

def train(epoch=3, log_steps=100):
    model.train()
    global_step = 0
    for epoch in range(epoch):
        for batch in trainset_loader:
            batch = {k: v.to('cuda') for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            if global_step % log_steps == 0:
                print(f"Epoch: {epoch}, Global Step: {global_step}, Loss: {loss.item()}")
            global_step += 1
        result = evaluate(valset_loader)
        print(f"Epoch: {epoch}, Val Result: {result}")

train(epoch=3, log_steps=100)

sentence = "这家餐厅的服务态度非常好，菜品也很新鲜。"
id2_label = {0: "差评", 1: "好评"}
with torch.no_grad():
    inputs = tokenizer(sentence, return_tensors="pt")
    inputs = {k: v.to('cuda') for k, v in inputs.items()}
    logits = model(**inputs).logits
    prediction = torch.argmax(logits, dim=-1).item()
    print(f"Sentence: {sentence}, Prediction: {prediction}")

model.config.id2label = id2_label
pipe_text_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)
print(pipe_text_classifier("这家餐厅的服务态度非常好，菜品也很新鲜。"))
