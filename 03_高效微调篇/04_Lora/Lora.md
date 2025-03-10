# LoRA

### 1. 导入库

```python
from peft import get_peft_model, TaskType, PromptTuningInit, PromptTuningConfig, PrefixTuningConfig, LoraConfig
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, \
    Trainer, TrainingArguments, AutoModelForCausalLM, pipeline
```

### 2. 加载数据集和分词器

```python
dataset = Dataset.load_from_disk('../../02_实战演练篇/09_对话机器人/alpaca_data_zh')
tokenizer = AutoTokenizer.from_pretrained("Langboat/bloom-1b4-zh")
```

### 3. 数据预处理

```python
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
```

### 4. 加载模型

```python
model = AutoModelForCausalLM.from_pretrained("Langboat/bloom-1b4-zh", low_cpu_mem_usage=True)
print(sum(param.numel() for param in model.parameters())) # 1303111680
print(model)
```

### 5. 配置 LoRA

```python
config = LoraConfig(task_type=TaskType.CAUSAL_LM, target_modules=".*\.1.*query_key_value", modules_to_save=["word_embeddings"]) # 正则表达式只应用于1和10~19层的query_key_value模块
model = get_peft_model(model, config)
print(model)
print(model.print_trainable_parameters())
```

### 6. 训练配置

```python
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

# trainer.train()
```

### 7. 预测

```python
pipeline = pipeline("text-generation",model=model, tokenizer=tokenizer, device=0)
ipt = "Human: {}\n{}".format("人工智能是什么", "").strip() + "\n\nAssistant: "
print(pipeline(ipt, max_length=64))
```

## 代码说明

### 数据预处理

`preprocess_function` 函数将数据集中的每个样本转换为模型输入格式。具体步骤包括：

1. 将指令和输入拼接成提示文本，并进行分词。
2. 将输出文本进行分词，并添加结束标记。
3. 将提示文本和输出文本的分词结果拼接，生成 `input_ids` 和 `attention_mask`。
4. 标签 `labels` 设置为 `-100` 掩码提示文本部分，只保留输出文本部分。
5. 如果长度超过最大长度 `max_length`，则进行截断。

### LoRA 配置

`LoraConfig` 用于配置 LoRA 微调参数，包括任务类型、目标模块和需要保存的模块。在本例中，LoRA 只应用于特定层的 `query_key_value` 模块。

### 训练配置

`TrainingArguments` 用于配置训练参数，包括输出目录、训练轮数、批量大小、梯度累积步数、日志步数和保存步数。

### 预测

使用 `pipeline` 进行文本生成，输入为提示文本，输出为生成的对话内容。