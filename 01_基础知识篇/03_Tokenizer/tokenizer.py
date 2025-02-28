"""
Author: Coder729
Date: 2025/2/26
Description: Tokenizer 分词器
"""
from transformers import AutoTokenizer

sentence = "我正在学习自然语言处理。" 
tokenizers = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-dianping-chinese") # 定义分词器
tokens = tokenizers.tokenize(sentence) # 分词
print("tokens:",tokens)

ids = tokenizers.convert_tokens_to_ids(tokens) # 将tokens转换为ids，每个token对应一个id
print("ids:",ids)

ids_encode = tokenizers.encode(sentence) # 直接转换成ids，会有[101]和[102]等特殊符号
print("ids_encode:",ids_encode)
print("tokens_encode:",tokenizers.convert_ids_to_tokens(ids)) # 将ids转换为tokens
print("ids_decode:",tokenizers.decode(ids_encode)) # 将ids转换为文本，会有[CLS]和[SEP]等特殊符号

print("限制最大长度:",tokenizers.tokenize(sentence,max_length=3)) # 限制最大长度
print("直接调用tokenizer:",tokenizers(sentence)) # tokenizer不仅显示ids，还显示attention mask，token type ids等信息

sentences = ["我正在学习自然语言处理。", "你好，世界！","自然语言处理是一门很有趣的学科。"]
print("批量分词:",tokenizers(sentences)) # 批量分词