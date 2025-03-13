"""
Author: Coder729
Date: 2025/3/13
Description: 
"""
import torch
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,AutoModel,
                          DataCollatorForSeq2Seq, TrainingArguments, Trainer)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

# 加载数据集
dataset = Dataset.load_from_disk("../../02_实战演练篇/09_对话机器人/alpaca_data_zh")
print(dataset[:3])

# 数据集预处理
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
print(tokenizer)
print(tokenizer(tokenizer.eos_token), tokenizer.eos_token_id)

def process_func(example):
    MAX_LENGTH = 256
    input_ids, attention_mask, labels = [], [], []
    instruction = "\n".join([example["instruction"], example["input"]]).strip()     # query
    instruction = tokenizer.build_chat_input(instruction, history=[], role="user")  # [gMASK]sop<|user|> \n query<|assistant|>
    response = tokenizer("\n" + example["output"], add_special_tokens=False)        # \n response, 缺少eos token
    input_ids = instruction["input_ids"][0].numpy().tolist() + response["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = instruction["attention_mask"][0].numpy().tolist() + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"][0].numpy().tolist()) + response["input_ids"] + [tokenizer.eos_token_id]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

tokenized_dataset = dataset.map(process_func, remove_columns=dataset.column_names)
print(tokenizer.decode(tokenized_dataset[1]["input_ids"]))
print(tokenizer.decode(list(filter(lambda x: x != -100, tokenized_dataset[1]["labels"]))))

# 创造模型
"""
新版本中需要将modeling_chatglm源码中的613行部分进行调整，代码如下：
```
if not kv_caches:
    kv_caches = [None for _ in range(self.num_layers)]
else:
    kv_caches = kv_caches[1]
```
如果不进行调整，后续chat阶段会报错
"""
model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True, torch_dtype=torch.bfloat16)
print(model)
for name, param in model.named_parameters():
    print(name)

config = LoraConfig(target_modules=["query_key_value"], modules_to_save=["post_attention_layernorm"])

model = get_peft_model(model, config)

for name, parameter in model.named_parameters():
    print(name)
model.print_trainable_parameters()

args = TrainingArguments(
    output_dir="./02_ChatGLM_半精度训练",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    logging_steps=10,
    num_train_epochs=1,
    learning_rate=1e-4,
    gradient_checkpointing=True,
    remove_unused_columns=False,
    save_steps=20,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset.select(range(6000)),
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()

# 推理
model.eval()
print(tokenizer.build_chat_input("考试的技巧有哪些？", history=[], role="user"))