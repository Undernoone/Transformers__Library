"""
Author: Coder729
Date: 2025/3/7
Description: 数据集：nlpcc_2017(和原作者的不一样)
             数据集原始版本features包括vision和data，data包括content和title两个字段。
             需要处理成只有content字段，title字段的格式。
             预训练模型：mengzi-t5-base
"""
import numpy as np
import torch
from rouge_chinese import Rouge
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, pipeline
from datasets import load_dataset, Dataset

from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("supremezxc/nlpcc_2017", split="train[:1%]") 
dataset = dataset.map(lambda example: {"title": example["data"]["title"],  "content": example["data"]["content"]}).remove_columns(["version", "data"]) # ，提出title和content字段，移除原始字段
dataset = dataset.select(range(min(5000, len(dataset))))
dataset = dataset.train_test_split(test_size=0.1)
tokenizer = AutoTokenizer.from_pretrained("Langboat/mengzi-t5-base")

# 3. 数据集处理
def preprocess_function(examples):
    contents = ["摘要生成：\n" + e for e in examples['content']]
    inputs = tokenizer(contents, max_length=512, truncation=True)
    labels = tokenizer(text_target=examples['title'], max_length=64, truncation=True)
    inputs['labels'] = labels['input_ids']
    return inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 4. 加载模型
model = AutoModelForSeq2SeqLM.from_pretrained("Langboat/mengzi-t5-base")

rouge = Rouge()

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = ["".join(p) for p in decoded_preds]
    decoded_labels = ["".join(l) for l in decoded_labels]
    scores = rouge.get_scores(decoded_preds, decoded_labels)
    return {"rouge1": scores["rouge-1"]["f"], "rouge2": scores["rouge-2"]["f"], "rougeL": scores["rouge-l"]["f"]}

training_args = Seq2SeqTrainingArguments(
    output_dir='./T5模型文本摘要',
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_checkpointing=8,
    logging_steps=8,
    evaluation_strategy="steps",
    save_strategy="steps",
    metric_for_best_model="rouge-1",
    prediction_loss_only=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 5. 训练模型
# trainer.train()

# 6. 评估模型
pipeline = pipeline('text2text-generation', model="./T5模型文本摘要/checkpoint-57", tokenizer=tokenizer, device=0, max_length=64, do_sample=True)
print(pipeline("摘要生成:\n" + dataset["test"][-1]["content"]))
print(dataset["test"][-1]["title"], end='')
# 自定义输入文本
sentence = "我在人工智能专业读硕士研究生，我正在学习NLP。"
print("原始语句:", sentence,"\n", pipeline("摘要生成::\n" +sentence))