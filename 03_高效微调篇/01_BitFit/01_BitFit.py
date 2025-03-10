"""
Author: Coder729
Date: 2025/3/9
Description: BitFit实战：只调节带bias的参数
"""

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, \
    Seq2SeqTrainingArguments, set_seed, DataCollatorForSeq2Seq, DataCollatorWithPadding, \
    Trainer, TrainingArguments, AutoModelForCausalLM, pipeline

dataset = Dataset.load_from_disk('../../02_实战演练篇/09_对话机器人/alpaca_data_zh')
dataset = dataset.select(range(min(500, len(dataset))))
tokenizer = AutoTokenizer.from_pretrained("Langboat/bloom-1b4-zh")
print(tokenizer)

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
print(model)
print(sum(param.numel() for param in model.parameters()))

# 只调节带bias的参数
num_param = 0
for name, param in model.named_parameters():
    if "bias" not in name:
        param.requires_grad = False
    else:
        num_param += param.numel()

print(num_param)

args = TrainingArguments(
    output_dir="./01_BitFit",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=8,
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()

# 预测
pipeline = pipeline("text-generation",model=model, tokenizer=tokenizer, device=0)
ipt = "Human: {}\n{}".format("怎么学习自然语言处理", "").strip() + "\n\nAssistant: "
print(pipeline(ipt, max_length=64))