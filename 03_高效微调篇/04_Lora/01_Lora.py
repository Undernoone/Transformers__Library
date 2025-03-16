"""
Author: Coder729
Date: 2025/3/10
Description: 
"""
import torch
from datasets import Dataset
from peft import get_peft_model, TaskType, LoraConfig, LoraModel
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, \
    Trainer, TrainingArguments, AutoModelForCausalLM, pipeline

dataset = Dataset.load_from_disk('../../02_实战演练篇/09_对话机器人/alpaca_data_zh')
dataset = dataset.select([i for i in range(1000)])
tokenizer = AutoTokenizer.from_pretrained("Langboat/bloom-1b4-zh")

def preprocess_function(examples):
    max_length = 256
    inputs_ids, attention_mask, labels = [], [], []
    instruction = tokenizer("\n".join(["Human: " + examples["instruction"], examples["input"]]).strip() + "\n\nAssistant: ")
    response = tokenizer(examples["output"] + tokenizer.eos_token)
    inputs_ids = instruction["input_ids"] + response["input_ids"]
    attention_mask = instruction["attention_mask"] + response["attention_mask"]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] # 模型只需要学习response
    if len(inputs_ids) > max_length:
        inputs_ids = inputs_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]
    return {"input_ids": inputs_ids, "attention_mask": attention_mask, "labels": labels}

tokenized_datasets = dataset.map(preprocess_function, remove_columns=dataset.column_names)

model = AutoModelForCausalLM.from_pretrained("Langboat/bloom-1b4-zh", low_cpu_mem_usage=True)
print(sum(param.numel() for param in model.parameters())) # 1303111680
print(model)

# Lora
config = LoraConfig(task_type=TaskType.CAUSAL_LM, target_modules=".*\.1.*query_key_value", modules_to_save=["word_embeddings"]) # 正则表达式只应用于1和10~19层的query_key_value模块
model = get_peft_model(model, config)
print(model)
print(model.print_trainable_parameters())

# 训练
args = TrainingArguments(
    output_dir="./Lora",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    logging_steps=10,
    save_steps=30,
)

trainer = Trainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
    train_dataset=tokenized_datasets,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()

# 在线预测
model = model.cuda()
input = tokenizer("Human: {}\n{}".format("人工智能是什么？", "").strip() + "\n\nAssistant: ", return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**input, max_length=128, do_sample=True)[0], skip_special_tokens=True))

# 离线预测
offline_model = AutoModelForCausalLM.from_pretrained("./Lora/checkpoint-125") # 需要自己看自己的checkpoint路径
offline_model = offline_model.to("cuda" if torch.cuda.is_available() else "cpu")
input_text = "Human: 人工智能是什么？\n\nAssistant: "
input_ids = tokenizer(input_text, return_tensors="pt").to(offline_model.device)
output = model.generate(**input_ids,max_length=128,do_sample=True, )
response = tokenizer.decode(output[0], skip_special_tokens=True)
print(response)