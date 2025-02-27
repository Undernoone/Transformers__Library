"""
Author: Coder729
Date: 2025/2/27
Description: AutoModel实战
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split
import evaluate
# class ClassificationDataset(Dataset):
#     def __init__(self) -> None:
#         super().__init__()
#         self.data = pd.read_csv('./datasets/ChnSentiCorp_htl_all.csv')
#         self.data = self.data.dropna() # 去掉空值
#     def __getitem__(self, index):
#         return self.data.iloc[index]["review"], self.data.iloc[index]["label"]
#     def __len__(self):
#         return len(self.data)
# 
# tokenizer = AutoTokenizer.from_pretrained("./rbt3")
# 
# # 划分数据集
# trainset, valset = random_split(ClassificationDataset(), lengths=[0.8, 0.2])
# print(len(trainset), len(valset))
# 
# # 定义数据加载器
# def collate_fn(batch):
#     texts, labels = [], []
#     for item in batch:
#         texts.append(item[0])
#         labels.append(item[1])
#     inputs = tokenizer(texts, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
#     inputs["labels"] = torch.tensor(labels)
#     return inputs
# 
# trainset_loader = DataLoader(trainset, batch_size=32, shuffle=False, collate_fn=collate_fn)
# valset_loader = DataLoader(valset, batch_size=32, shuffle=False, collate_fn=collate_fn)
# 
# model = AutoModelForSequenceClassification.from_pretrained("./rbt3").cuda()
# 
# optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
# 
# def train(epoch=3,log_steps=100):
#     global_step = 0
#     for epoch in range(epoch):
#         for batch in trainset_loader:
#             if torch.cuda.is_available():
#                 batch = {k: v.cuda() for k, v in batch.items()}
#             optimizer.zero_grad()
#             outputs = model(**batch)
#             loss = outputs.loss
#             loss.backward()
#             optimizer.step()
#             if global_step % log_steps == 0:
#                 print(f"Epoch: {epoch}, Global Step: {global_step}, Loss: {loss.item()}")
#             global_step += 1
#     print("Training Done!")
# 
# def evaluate(loader):
#     model.eval()
#     total_loss = 0
#     total_correct = 0
#     with torch.no_grad():
#         for batch in loader:
#             if torch.cuda.is_available():
#                 batch = {k: v.cuda() for k, v in batch.items()}
#             outputs = model(**batch)
#             loss = outputs.loss
#             total_loss += loss.item()
#             predictions = torch.argmax(outputs.logits, dim=-1)
#             total_correct += (predictions == batch["labels"]).sum().item()
#     return total_loss / len(loader), total_correct / len(loader.dataset)
# 
# train(epoch=3, log_steps=100)
# 
# sentence = "这家餐厅的服务态度非常好，菜品也很新鲜。"
# id2_label = {0: "差评", 1: "好评"}
# with torch.inference_mode():
#     inputs = tokenizer(sentence,return_tensors="pt")
#     inputs = {k: v.cuda() for k, v in inputs.items()}
#     logits = model(**inputs).logits
#     prediction = torch.argmax(logits, dim=-1).item()
#     print(f"Sentence: {sentence}, Prediction: {prediction}")
# 
# model.config.id2label = id2_label
# pipe_text_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)
# print(pipe_text_classifier("这家餐厅的服务态度非常好，菜品也很新鲜。"))


class ClassificationDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.data = pd.read_csv('./datasets/ChnSentiCorp_htl_all.csv')
        self.data = self.data.dropna() # 去掉空值
    def __getitem__(self, index):
        return self.data.iloc[index]["review"], self.data.iloc[index]["label"]
    def __len__(self):
        return len(self.data)

tokenizer = AutoTokenizer.from_pretrained("./rbt3")

# 划分数据集
trainset, valset = random_split(ClassificationDataset(), lengths=[0.8, 0.2])
print(len(trainset), len(valset))

# 定义数据加载器
def collate_fn(batch):
    texts, labels = [], []
    for item in batch:
        texts.append(item[0])
        labels.append(item[1])
    inputs = tokenizer(texts, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    inputs["labels"] = torch.tensor(labels)
    return inputs

trainset_loader = DataLoader(trainset, batch_size=32, shuffle=False, collate_fn=collate_fn)
valset_loader = DataLoader(valset, batch_size=32, shuffle=False, collate_fn=collate_fn)

model = AutoModelForSequenceClassification.from_pretrained("./rbt3").cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
classification_matrix_实战 = evaluate.combine(["accuracy", "f1"])

def evaluate(loader):
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1).cuda()
            classification_matrix_实战.add_batch(predictions=predictions.long(), references=batch["labels"].long())
    return classification_matrix_实战.compute()

def train(epoch=3,log_steps=100):
    model.train()
    global_step = 0
    for epoch in range(epoch):
        for batch in trainset_loader:
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
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

