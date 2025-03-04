"""
Author: Coder729
Date: 2025/3/4
Description: 使用回归方法训练文本相似度，因为很多时候不是一对一的相似度匹配，而是一对多的相似度匹配，
             因此这就相当于多项选择，把一当作问题，多当作选项，因此需要用回归的方法来训练。
"""

import torch
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, \
    DataCollatorWithPadding, pipeline

datasets = load_dataset("json", data_files="./train_pair_1w.json", split="train")
datasets = datasets.train_test_split(test_size=0.2)

tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")
print(datasets["train"][0]) # 可以看到包括sentence1, sentence2, label三个字符串字段

def preprocess_function(examples):
    tokenized_examples = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True,max_length=128)
    tokenized_examples["label"] = [float(label) for label in examples["label"]] # 神经网络训练时不能用字符串，所以需要转换为整数
    return tokenized_examples

tokenized_datasets = datasets.map(preprocess_function, batched=True, remove_columns=datasets["train"].column_names)
# print(type(tokenized_datasets["train"][0]["label"]))

model = AutoModelForSequenceClassification.from_pretrained("hfl/chinese-macbert-base",num_labels=1) # 如果num_labels=1, model会自动识别出这是一个回归任务，自动使用MSELoss作为损失函数

acc_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def eval_metric(eval_predict):
    predictions, labels = eval_predict
    predictions = [int(p)>0.5 for p in predictions]
    labels = [int(l) for l in labels]
    predictions = torch.tensor(predictions)
    # predictions = torch.argmax(predictions, axis=-1)
    acc = acc_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    acc.update(f1)
    return acc

training_args = TrainingArguments(
    output_dir='./文本相似度',
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=1,
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    metric_for_best_model="f1",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=eval_metric)

trainer.train()
trainer.evaluate(tokenized_datasets["test"])

