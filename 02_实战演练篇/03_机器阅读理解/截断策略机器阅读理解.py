"""
Author: Coder729
Date: 2025/3/2
Description: 阶段策略机器阅读理解
"""
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering,TrainingArguments, Trainer, DefaultDataCollator

datasets = load_dataset("hfl/cmrc2018") # 机器阅读理解数据集，包括id、context、question、answers
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")

"""
在 transformers 库的 AutoTokenizer 中，text 和 text_pair 不是默认参数，
而是 tokenizer 方法的关键字参数（keyword arguments）。它们的作用如下：
text: 主要输入文本（通常是问题 question）。
text_pair: 可选的辅助输入（通常是上下文 context）。
"""

def process_function(examples):
    tokenized_examples = tokenizer(text=examples["question"],
                                   text_pair=examples["context"], 
                                   # 输入文本为question和context, tokenizer会自动将question和context拼接起来，在他们的input_ids中插入分割符
                                   max_length=512,
                                   truncation="only_second", # 只截断context，不截断question
                                   padding="max_length",
                                   return_offsets_mapping=True # 字符索引映射
                                   )
    offset_mapping = tokenized_examples.pop("offset_mapping") # 提取出offset_mapping
    start_positions = []
    end_positions = []
    for index, offsets in enumerate(offset_mapping):
        answer = examples["answers"][index]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])
        context_start = tokenized_examples.sequence_ids(index).index(1)
        context_end = tokenized_examples.sequence_ids(index).index(None, context_start)-1
        # 判断答案是否在context的有效范围内
        if offsets[context_end][1] < start_char or offsets[context_start][0] > end_char:
            start_token_pos = 0
            end_token_pos = 0
        else:
            token_id = context_start
            while token_id <= context_end and offsets[token_id][0] < start_char:
                token_id += 1
            start_token_pos = token_id
            token_id = context_end
            while token_id >= context_start and offsets[token_id][1] > end_char:
                token_id -= 1
            end_token_pos = token_id
        start_positions.append(start_token_pos)
        end_positions.append(end_token_pos)
    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions
    return tokenized_examples

tokenized_datasets = datasets.map(process_function, batched=True, remove_columns=datasets["train"].column_names)
print(tokenized_datasets)

model = AutoModelForQuestionAnswering.from_pretrained("hfl/chinese-macbert-base")

args = TrainingArguments(
    output_dir="model_for_qa",
    per_device_train_batch_size=32,
    gradient_accumulation_steps=32,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    num_train_epochs=2
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=DefaultDataCollator()
)

trainer.train()