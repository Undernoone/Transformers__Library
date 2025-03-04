"""
Author: Coder729
Date: 2025/3/3
Description: 
"""
from typing import Any

import torch
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMultipleChoice, Trainer, TrainingArguments

dataset = load_dataset("clue/clue", "c3",)
dataset.pop("test")
# 为了快速演示，我们只使用训练集的前100条数据和验证集的前50条数据,可以自行修改
dataset["train"] = dataset["train"].select(range(100))
dataset["validation"] = dataset["validation"].select(range(50))

tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")

def preprocess_function(examples): # examples是一个list，包括id、context、question、choice、answer，目的是把这个list在tokenizer除了基础的还要添加一个labels

    context = []
    questions_choices = []
    labels = []
    """
    labels是需要把文本定位成是选项中的第几个,例如dataset["train"][0]中的
    'choice': ['恐怖片', '爱情片', '喜剧片', '科幻片'], 'answer': '喜剧片'
    需要把喜剧片定位到choice中的3
    """
    for index in range(len(examples["context"])):
        ctx = "\n".join(examples["context"][index])
        question = examples["question"][index]
        choices = examples["choice"][index]
        for choice in choices:
            context.append(ctx)
            questions_choices.append(question + " " + choice)
        if len(choices) < 4:
            for _ in range(4-len(choices)):
                context.append(ctx)
                questions_choices.append(question + " " + "不知道")
        labels.append(choices.index(examples["answer"][index]))
    tokenized_examples = tokenizer(context, questions_choices, padding="max_length", max_length=256, truncation="only_first")
    tokenized_examples = {k: [v[i:i+4] for i in range(0, len(v), 4)]for k, v in tokenized_examples.items()}
    tokenized_examples["labels"] = labels
    return tokenized_examples

tokenized_datasets = dataset.map(preprocess_function, batched=True)
print(tokenized_datasets)

model = AutoModelForMultipleChoice.from_pretrained("hfl/chinese-macbert-base")

accuracy = evaluate.load("accuracy")

def compute_metrics(pred):
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="./多项选择模型",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,
    )

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics
)

trainer.train()

# 无法使用pipeline
class MultipleChoicePipeline:

    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device

    def preprocess(self, context, quesiton, choices):
        cs, qcs = [], []
        for choice in choices:
            cs.append(context)
            qcs.append(quesiton + " " + choice)
        return tokenizer(cs, qcs, truncation="only_first", max_length=256, return_tensors="pt")

    def predict(self, inputs):
        inputs = {k: v.unsqueeze(0).to(self.device) for k, v in inputs.items()}
        return self.model(**inputs).logits

    def postprocess(self, logits, choices):
        predition = torch.argmax(logits, dim=-1).cpu().item()
        return choices[predition]

    def __call__(self, context, question, choices) -> Any:
        inputs = self.preprocess(context, question, choices)
        logits = self.predict(inputs)
        result = self.postprocess(logits, choices)
        return result

pipe = MultipleChoicePipeline(model, tokenizer)

print(pipe("小明在北京上班", "小明在哪里上班？", ["北京", "上海", "河北", "海南", "河北", "海南"]))