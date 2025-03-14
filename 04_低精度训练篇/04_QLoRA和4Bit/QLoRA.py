"""
Author: Coder729
Date: 2025/3/14
Description: 
"""
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer

dataset = Dataset.load_from_disk("../../02_实战演练篇/09_对话机器人/alpaca_data_zh")
print(dataset)
tokenizer = AutoTokenizer.from_pretrained("D:/Study_Date/Modelscope/cache/modelscope/Llama-2-7b-ms")
tokenizer.padding_side = "right" # Llama默认设置是left，需要改成right, 不然有时候出问题不收敛
tokenizer.pad_token_id = 2

def preprocess_function(examples):
    max_length = 1024 # Llama的分词器对英文适配不好，如果还设置成256，可能会导致截断
    inputs_ids, attention_mask, labels = [], [], []
    instruction = tokenizer("\n".join(["Human: " + examples["instruction"], examples["input"]]).strip() + "\n\nAssistant: ",add_special_tokens=False)
    response = tokenizer(examples["output"], add_special_tokens=False)
    inputs_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.eos_token_id] # 模型只需要学习response
    if len(inputs_ids) > max_length:
        inputs_ids = inputs_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]
    return {"input_ids": inputs_ids, "attention_mask": attention_mask, "labels": labels}

tokenized_datasets = dataset.map(preprocess_function,remove_columns=dataset.column_names)

# 加载模型
model = AutoModelForCausalLM.from_pretrained("D:/Study_Date/Modelscope/cache/modelscope/Llama-2-7b-ms", low_cpu_mem_usage=True,
                                             torch_dtype=torch.bfloat16, device_map="auto", load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
for name, param in model.named_parameters():
    print(name, param.dtype) # 可以看到当前所有层都是float16

config = LoraConfig(task_type=TaskType.CAUSAL_LM)
model = get_peft_model(model, config)
model.enable_input_require_grads()# 开启梯度检查点时候必须要开启这个选项

# 训练
args = TrainingArguments(
    output_dir="./QLoRA",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    logging_steps=10,
    save_steps=20,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit"
)

trainer = Trainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
    train_dataset=tokenized_datasets,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
trainer.train()

# model.eval()

ipt = tokenizer("Human: {}\n{}".format("人工智能是什么？", "").strip() + "\n\nAssistant: ", return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**ipt, max_length=512, do_sample=True, eos_token_id=tokenizer.eos_token_id)[0], skip_special_tokens=True))