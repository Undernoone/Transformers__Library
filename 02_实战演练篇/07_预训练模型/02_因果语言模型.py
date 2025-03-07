"""
Author: Coder729
Date: 2025/3/7
Description: 因果语言模型，自回归模型：
             将完整序列输入，基于上文的token取预测当前token
             结束位置要有特殊token:eos_token
             数据集：wikipedia-cn-20230720-fitered
             tokenizer: Langboat/bloom-389m-zh
"""

from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments, pipeline

# 1. 加载数据集
datasets = Dataset.load_from_disk('./wiki_cn_filtered')
print(datasets) # 这个数据集有两个column，一个是source，一个是completion
datasets = datasets.select(range(300)) # 快速测试时使用

# 2. 数据集预处理
tokenizer = AutoTokenizer.from_pretrained("Langboat/bloom-389m-zh")
def preprocess_function(examples):
    # 在数据集中的completion后面加上eos_token，eos_token表示句子结束
    contents = [completion + tokenizer.eos_token for completion in examples["completion"]]
    return tokenizer(contents, max_length=512, truncation=True)

tokenized_datasets = datasets.map(preprocess_function, batched=True, remove_columns=datasets.column_names) # source、completion ----> input_ids、attention_mask
dataloader = DataLoader(tokenized_datasets, batch_size=2,
                        collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)) # mlm=True表示使用Masked Language Modeling

# 3. 定义训练模型
model = AutoModelForCausalLM.from_pretrained("Langboat/bloom-389m-zh")
training_args = TrainingArguments(
    output_dir='./因果语言模型',
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    logging_steps=10,
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    train_dataset=tokenized_datasets,
)

# 4. 训练模型
trainer.train()

# 5. 模型推理
pipe = pipeline("text-generation", model='./因果语言模型/checkpoint-9', tokenizer=tokenizer, device=0) # 可以model直接写model，也可以写训练好的checkpoint路径
print(pipe("西安交通大学博物馆（Xi'an Jiaotong University Museum）是一座位于西安", max_length=128, do_sample=True))
print(pipe("下面是一则游戏新闻。小编报道，近日，游戏产业发展的非常", max_length=128, do_sample=True))


