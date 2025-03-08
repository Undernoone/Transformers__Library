"""
Author: Coder729
Date: 2025/3/8
Description: 
"""

import numpy as np
import torch
from rouge_chinese import Rouge
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, \
    Seq2SeqTrainer, pipeline, AutoModel
from datasets import load_dataset
from transformers import AutoTokenizer
dataset = load_dataset("supremezxc/nlpcc_2017", split="train[:1%]")

dataset = dataset.map(lambda example: {"title": example["data"]["title"],  "content": example["data"]["content"]}).remove_columns(["version", "data"]) # ，提出title和content字段，移除原始字段
dataset = dataset.select(range(min(5000, len(dataset))))
dataset = dataset.train_test_split(test_size=0.1)
tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-large-chinese", trust_remote_code=True)# GLM并没被发布到huggingface，所以需要trust_remote_code=True

# 3. 数据集处理
def preprocess_function(examples):
    contents = ["摘要生成：\n" + e + tokenizer.eos_token for e in examples['content']]
    inputs = tokenizer(contents, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
    inputs = tokenizer.build_inputs_with_special_tokens(inputs, target=examples['title'], padding='max_length', max_length=64)
    return inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

# 4. 加载模型
model = AutoModel.from_pretrained("THUDM/glm-large-chinese", trust_remote_code=True)
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
    output_dir='./GLM模型文本摘要',
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_checkpointing=8,
    logging_steps=8,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    tokenizer=tokenizer,
)

# 5. 训练模型
trainer.train()

# 6. 评估模型
pipeline = pipeline('text2text-generation', model="model", tokenizer=tokenizer, device=0, max_length=64, do_sample=True)
print(pipeline("摘要生成:\n" + dataset["test"][-1]["content"]))
print(dataset["test"][-1]["title"], end='')
# 自定义输入文本
sentence = "我在人工智能专业读硕士研究生，我正在学习NLP。"
print("原始语句:", sentence,"\n", pipeline("摘要生成::\n" +sentence))