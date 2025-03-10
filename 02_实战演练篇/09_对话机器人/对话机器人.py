"""
Author: Coder729
Date: 2025/3/8
Description: 原始数据集有三条键值对，分别是：input, output, instruction。
             部分数据没有input。
"""
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments, AutoModelForCausalLM, pipeline

dataset = Dataset.load_from_disk('./alpaca_data_zh')
dataset = dataset.select(range(min(500, len(dataset))))
print(dataset[0])
tokenizer = AutoTokenizer.from_pretrained("Langboat/bloom-389m-zh")

# 数据预处理
# 把数据处理成Human[instrustion] + [input](可以没有） + Assistant: [output]
def preprocess_function(examples):
    max_length = 256
    inputs_ids, attention_mask, labels = [], [], []
    instruction = tokenizer("\n".join(["Human: " + examples["instruction"], examples["input"]]).strip() + "\n\nAssistant: ")
    response = tokenizer(examples["output"] + tokenizer.eos_token)
    print(tokenizer.decode(instruction["input_ids"] + response["input_ids"]))
    inputs_ids = instruction["input_ids"] + response["input_ids"]
    attention_mask = instruction["attention_mask"] + response["attention_mask"]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] # 模型只需要学习response
    if len(inputs_ids) > max_length:
        inputs_ids = inputs_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]
    return {"input_ids": inputs_ids, "attention_mask": attention_mask, "labels": labels}

tokenized_datasets = dataset.map(preprocess_function, remove_columns=dataset.column_names)

model = AutoModelForCausalLM.from_pretrained("Langboat/bloom-389m-zh")

args = TrainingArguments(
    output_dir="./对话机器人",
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

# trainer.train()

# 预测
pipeline = pipeline("text-generation",model='对话机器人/checkpoint-4', tokenizer=tokenizer, device=0)
ipt = "Human: {}\n{}".format("怎么学习自然语言处理", "").strip() + "\n\nAssistant: "
print(pipeline(ipt, max_length=64))