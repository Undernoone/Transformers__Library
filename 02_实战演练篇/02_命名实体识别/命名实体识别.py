"""
Author: Coder729
Date: 2025/2/28
Description: 命名实体识别(NER)
"""
import seqeval
import evaluate
import numpy as np
from pprint import pprint
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer , DataCollatorForTokenClassification, pipeline

# 加载数据集
dataset = load_dataset("peoples_daily_ner", trust_remote_code=True) # 人民日报数据集

"""
pprint(dataset['train'][0])：输出数据集的第一个数据，但是不知道其中的特征含义，
例如这里的[0, 0, 0, 0, 0, 0, 0, 5, 6, 0, 5, 6, 0, 0, 0, 0, 0, 0]不知道是什么意思，使用features属性可以查看每个特征对应的含义↓
pprint(dataset['train'].features)：可以看到['O','B-PER','I-PER','B-ORG','I-ORG','B-LOC','I-LOC']，可以发现5对应B-LOC，6对应I-LOC。
"""

# 因为要做命名实体识别，所以需要取出ner_tags这个特征，并将其转换为标签。
ner_labels_list = dataset['train'].features['ner_tags'].feature.names
print("ner_labels_list:",ner_labels_list)
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")
print(tokenizer(dataset['train'][0]['tokens'],is_split_into_words=True)) # 如果不加is_split_into_words=True，则会将划分好的句子再次划分。

"""
sentences = "I am studying NLP."
print(tokenizer(sentences)) 
# 输出结果为[101, 151, 8413, 12685, 8221, 156, 10986, 119, 102]，四个单词却被分成了七个token，所以需要找到每个token对应的单词。
print(tokenizer(sentences).word_ids()) # tokenizer.word_ids():[None, 0, 1, 2, 2, 3, 3, 4, None],可以查看每个token对应的单词的id。
"""

# 所以需要写一个处理函数，把ids和labels对应起来。
def process_function(example: dict):
    tokenized_example = tokenizer(example['tokens'],max_length=128,truncation=True,is_split_into_words=True) # 拆词
    labels = []
    for i,label in enumerate(example['ner_tags']):
        word_ids = tokenized_example.word_ids(batch_index=i) # 获取每个token对应的单词的id
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100) # -100在交叉熵cross-entropy中代表padding
            else:
                label_ids.append(label[word_id]) # 向tokenizer_example遍历添加labels
        labels.append(label_ids)
    tokenized_example['labels'] = labels
    return tokenized_example

tokenized_datasets = dataset.map(process_function, batched=True)
print(tokenized_datasets)

model = AutoModelForTokenClassification.from_pretrained("hfl/chinese-macbert-base",num_labels=len(ner_labels_list)) # 默认是二分类，所以这里改成多分类。

seqeval = evaluate.load("seqeval")

def eval_metric(pred):
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=-1)
    true_predictions = [
        [ner_labels_list[p] for p, l in zip(pred_seq, label_seq) if l != -100]
        for pred_seq, label_seq in zip(predictions, labels)
    ]
    true_labels = [
        [ner_labels_list[l] for p, l in zip(pred_seq, label_seq) if l != -100]
        for pred_seq, label_seq in zip(predictions, labels)
    ]

    result = seqeval.compute(predictions=true_predictions, references=true_labels, mode="strict", scheme="IOB2")
    return {"f1": result["overall_f1"]}


args = TrainingArguments(
    output_dir='./model_for_ner',
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    eval_strategy="epoch",
    save_strategy='epoch',
    metric_for_best_model='f1',
    load_best_model_at_end=True,
    logging_steps=50,
    num_train_epochs=1
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    compute_metrics=eval_metric,
    data_collator=DataCollatorForTokenClassification(tokenizer)
)

trainer.train()
trainer.evaluate(eval_dataset=tokenized_datasets["test"])

# 预测
model.config.id2label = {idx: label for idx, label in enumerate(ner_labels_list)}
ner_pipeline = pipeline('token-classification', model=model, tokenizer=tokenizer, device=0, aggregation_strategy="simple")
pprint(ner_pipeline("小明正在长春学习自然语言处理。"),width=100)
