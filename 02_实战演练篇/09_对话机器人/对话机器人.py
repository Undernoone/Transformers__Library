"""
Author: Coder729
Date: 2025/3/2
Description: 
"""
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer,\
    Seq2SeqTrainingArguments, set_seed, DataCollatorForSeq2Seq, DataCollatorWithPadding, \
    Trainer, TrainingArguments

dataset = Dataset.load_from_disk('./alpaca_data_zh')
print(dataset[0])

tokenizer = AutoTokenizer.from_pretrained("Langboat/bloom-389m-zh")
print(tokenizer)