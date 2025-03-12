"""
Author: Coder729
Date: 2025/3/12
Description: 
"""
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments

dataset = Dataset.load_from_disk("../../02_实战演练篇/09_对话机器人/alpaca_data_zh")
# dataset
# dataset[0:3]
tokenizer = AutoTokenizer.from_pretrained("D:/Study_Date/Modelscope/cache/modelscope/Llama-2-7b-ms")
# tokenizer
def preprocess_function(examples):
    max_length = 384
    inputs_ids, attention_mask, labels = [], [], []
    instruction = tokenizer("\n".join(["Human: " + examples["instruction"], examples["input"]]).strip() + "\n\nAssistant: ",add_special_tokens=False)
    response = tokenizer(examples["output"] + tokenizer.eos_token)
    inputs_ids = instruction["input_ids"] + response["input_ids"]
    attention_mask = instruction["attention_mask"] + response["attention_mask"]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] # 模型只需要学习response
    if len(inputs_ids) > max_length:
        inputs_ids = inputs_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]
    return {"input_ids": inputs_ids, "attention_mask": attention_mask, "labels": labels}
tokenized_datasets = dataset.map(preprocess_function,remove_columns=dataset.column_names)
# tokenized_datasets
# tokenizer.decode(tokenized_datasets[0]["input_ids"])
model = AutoModelForCausalLM.from_pretrained("D:/Study_Date/Modelscope/cache/modelscope/Llama-2-7b-ms",low_cpu_mem_usage=True,torch_dtype=torch.float16,device_map="auto")

config = LoraConfig(task_type=TaskType.CAUSAL_LM, target_modules=".*\.1.*query_key_value", modules_to_save=["word_embeddings"]) # 正则表达式只应用于1和10~19层的query_key_value模块
model = get_peft_model(model, config)
# print(model)
print(model.print_trainable_parameters())