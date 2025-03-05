"""
Author: Coder729
Date: 2025/3/5
Description: 
"""
import tqdm
import torch
import evaluate
import pandas as pd
from typing import Optional
from datasets import load_dataset
from torch.nn import CosineSimilarity, CosineEmbeddingLoss
from transformers.configuration_utils import PretrainedConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, \
    DataCollatorWithPadding, pipeline, BertPreTrainedModel, PreTrainedTokenizer, BertModel
# 输出tqdm库的位置
data = pd.read_csv('law_faq.csv')
print(data.head())

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

model = DualModel.from_pretrained("D:\Study_Date\LLMs_study\Transformers库\\02_实战演练篇\\05_文本相似度\向量匹配策略解决大候选样本的解决方案\checkpoint-250")
model = model.cuda()
model.eval()

tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")

# 将Batch转化成向量

