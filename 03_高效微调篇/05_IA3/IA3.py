"""
Author: Coder729
Date: 2025/3/11
Description: 
"""
from datasets import Dataset
from peft import get_peft_model, TaskType, IA3Config
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, \
    Trainer, TrainingArguments, AutoModelForCausalLM, pipeline

dataset = Dataset.load_from_disk('../../02_实战演练篇/09_对话机器人/alpaca_data_zh')
dataset = dataset.select([i for i in range(5000)]) # 取1000条数据进行训练
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
config = IA3Config(task_type=TaskType.CAUSAL_LM)
model = get_peft_model(model, config)
print(model)
'''
可以看到(base_layer): Linear(in_features=2048, out_features=6144, bias=True) ----->
(lora_A): ModuleDict((default): Linear(in_features=2048, out_features=8, bias=False))
(lora_B): ModuleDict((default): Linear(in_features=8, out_features=6144, bias=False))
'''
print(model.print_trainable_parameters())

# 训练
args = TrainingArguments(
    output_dir="./Lora",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    logging_steps=10,
    save_steps=1,
)

trainer = Trainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
    train_dataset=tokenized_datasets,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()

# 预测
pipeline = pipeline("text-generation",model=model, tokenizer=tokenizer, device=0)
ipt = "Human: {}\n{}".format("人工智能是什么", "").strip() + "\n\nAssistant: "
print(pipeline(ipt, max_length=64))