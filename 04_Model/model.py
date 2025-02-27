"""
Author: Coder729
Date: 2025/2/27
Description: AutoModel的使用
"""
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers import BertModel, BertTokenizer, BertConfig
from pprint import pprint
from transformers import AutoModelForSequenceClassification

model = AutoModel.from_pretrained("./rbt3") # 加载预训练模型，这里使用的是Huggingface的rbt3模型，可以使用远程自动下载也可以本地下载
print("Model config:", model.config) # 可以查看模型的配置信息

config = AutoConfig.from_pretrained("./rbt3") # 加载模型配置信息，之后便可以修改模型的配置

sentence = "I am studying NLP."

tokenizer = AutoTokenizer.from_pretrained("./rbt3")

inputs = tokenizer(sentence, return_tensors="pt")
pprint(inputs, width=100)
outputs = model(**inputs)
pprint(outputs, width=100)

model_with_attentions = AutoModel.from_pretrained("./rbt3", output_attentions=True) # 可视化attention weights
inputs = tokenizer(sentence, return_tensors="pt")
outputs = model_with_attentions(**inputs)
pprint(outputs, width=100)

