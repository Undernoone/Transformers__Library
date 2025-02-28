"""
Author: Coder729
Date: 2025/2/28
Description: 降低显存占用实战
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, \
    DataCollatorWithPadding
from datasets import load_dataset
import torch
import evaluate

# 加载数据集
datasets = load_dataset("csv", data_files="../../01_基础知识篇/04_Model/datasets/ChnSentiCorp_htl_all.csv", split="train")
datasets = datasets.filter(lambda x: x["review"] is not None)
print(datasets)
datasets = datasets.train_test_split(test_size=0.1) # 将数据集划分成train和test两部分
print(datasets)


# 定义预处理函数
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-large")

def preprocess_function(examples):
    tokenized_examples = tokenizer(examples["review"], max_length=32, padding="max_length", truncation=True)
    tokenized_examples["label"] = examples["label"]
    return tokenized_examples

tokenized_datasets = datasets.map(preprocess_function, batched=True, remove_columns=datasets["train"].column_names)
print(tokenized_datasets)

model = AutoModelForSequenceClassification.from_pretrained("hfl/chinese-macbert-large")

# 加载评价指标
acc_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")


# 定义评估指标函数
def eval_metric(eval_predict):
    predictions, labels = eval_predict
    predictions = torch.argmax(predictions, axis=-1)
    acc = acc_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    acc.update(f1)
    return acc

tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-large")

"""
降低显存占用的多种方法，具体方法参考官方文档https://huggingface.co/docs/transformers/perf_train_gpu_one
1. 减少 batch size
2. 开启梯度检查点
3. 优化器选择 adafactor
4. 冻结部分层的梯度使显存占用降低
"""

# 默认配置
training_args = TrainingArguments(
    output_dir='./checkpoint',
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

training_args_2 = TrainingArguments(
    output_dir='./checkpoint',
    per_device_train_batch_size=1, # 将 batch size 设为 1
    gradient_accumulation_steps=32,
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

training_args_3 = TrainingArguments(
    output_dir='./checkpoint',
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    gradient_checkpointing=True, # 开启梯度检查点，官方文档https://huggingface.co/docs/transformers/perf_train_gpu_one#gradient-checkpointing
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
training_args_4 = TrainingArguments(
    output_dir='./checkpoint',
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    gradient_checkpointing=True,
    optim="adafactor", # 修改优化器为 adafactor
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
print(training_args)

# 冻结部分层的梯度使显存占用降低
for name, param in model.bert.named_parameters():
    param.requires_grad = False

trainer = Trainer(model=model,
                  args=training_args,
                  train_dataset=tokenized_datasets["train"],
                  eval_dataset=tokenized_datasets["test"],
                  data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                  compute_metrics=eval_metric)
trainer.train()