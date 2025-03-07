"""
Author: Coder729
Date: 2025/3/5
Description: 向量匹配策略解决大候选样本的解决方案：
             回归方法每次都要于所有候选样本计算相似度，计算量过大，效率低下。
"""

import torch
import evaluate
from typing import Optional
from datasets import load_dataset
from torch.nn import CosineSimilarity, CosineEmbeddingLoss
from transformers.configuration_utils import PretrainedConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, \
    DataCollatorWithPadding, pipeline, BertPreTrainedModel, PreTrainedTokenizer, BertModel

datasets = load_dataset("json", data_files="./train_pair_1w.json", split="train")
datasets = datasets.train_test_split(test_size=0.2)
print(datasets)
# datasets["train"] = datasets["train"].select(range(100)) # 快速测试时使用
# datasets["test"] = datasets["test"].select(range(10))
# print(datasets)

tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")
# print(datasets["train"][0]) # 可以看到包括sentence1, sentence2, label三个字符串字段
print(tokenizer)

def preprocess_function(examples):
    sentences = []
    labels = []
    for sen1, sen2, label in zip(examples["sentence1"], examples["sentence2"], examples["label"]):
        sentences.append(sen1)
        sentences.append(sen2)
        labels.append(1 if int(label) > 1 else -1)
    tokenized_examples = tokenizer(sentences, padding="max_length", truncation=True, max_length=128)
    tokenized_examples = {k: [v[i:i+2] for i in range(0, len(v), 2)] for k, v in tokenized_examples.items()}
    tokenized_examples["label"] = labels
    return tokenized_examples

tokenized_datasets = datasets.map(preprocess_function, batched=True, remove_columns=datasets["train"].column_names)
# print(tokenized_datasets["train"][0])
# print(type(tokenized_datasets["train"][0]["label"]))

# model = AutoModelForSequenceClassification.from_pretrained("hfl/chinese-macbert-base",num_labels=1) # 如果num_labels=1, model会自动识别出这是一个回归任务，自动使用MSELoss作为损失函数

class DualModel(BertPreTrainedModel):

    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.bert = BertModel(config)
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        senA_input_ids, senB_input_ids = input_ids[:, 0], input_ids[:, 1]
        senA_attention_mask, senB_attention_mask = attention_mask[:, 0], attention_mask[:, 1]
        senA_token_type_ids, senB_token_type_ids = token_type_ids[:, 0], token_type_ids[:, 1]

        senA_outputs = self.bert(
            senA_input_ids,
            attention_mask=senA_attention_mask,
            token_type_ids=senA_token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        senA_pooled_output = senA_outputs[1]
        senB_outputs = self.bert(
            senB_input_ids,
            attention_mask=senB_attention_mask,
            token_type_ids=senB_token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        senB_pooled_output = senB_outputs[1]
        cos = CosineSimilarity()(senA_pooled_output, senB_pooled_output)
        loss = None
        if labels is not None:
            loss_fct = CosineEmbeddingLoss(0.3)
            loss = loss_fct(senA_pooled_output, senB_pooled_output, labels)
        output = (cos,)
        return ((loss,) + output) if loss is not None else output

# model = DualModel.from_pretrained("D:\Study_Date\HuggingFace\cache\models--hfl--chinese-macbert-base")
model = DualModel.from_pretrained("hfl/chinese-macbert-base", num_labels=1)
acc_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def eval_metric(eval_predict):
    predictions, labels = eval_predict
    predictions = [int(p > 0.7) for p in predictions]
    labels = [int(l > 0) for l in labels]
    # predictions = predictions.argmax(axis=-1)
    acc = acc_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    acc.update(f1)
    return acc

training_args = TrainingArguments(
    output_dir='./回归方法过大时解决方案',
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=1,
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    metric_for_best_model="f1",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=eval_metric)

trainer.train()
trainer.evaluate(tokenized_datasets["test"])