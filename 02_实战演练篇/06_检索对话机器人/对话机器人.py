"""
Author: Coder729
Date: 2025/3/5
Description: 检索对话机器人
"""
import faiss
import torch
import pandas as pd
from tqdm import tqdm
from DualModel import DualModel
from transformers import AutoTokenizer, BertForSequenceClassification

# 1. 加载数据集
data = pd.read_csv('law_faq.csv') # title作为问题，reply作为回答

# 2. 加载模型和tokenizer (这里使用的是05_文本相似度中训练好的模型权重）
model = DualModel.from_pretrained("../05_文本相似度/向量匹配策略解决大候选样本的解决方案/checkpoint-250")
model = model.cuda()
model.eval()
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")

# 3. 将问题编码成向量
# 检索对话机器人需要将我们提问的问题和知识库中的问题进行相似度匹配，如果每次都是实时匹配则会耗费大量时间，所以提前将问题转化为向量
questions = data['title'].to_list()
questions_vectors = []
with torch.inference_mode():
    for i in tqdm(range(0, len(questions), 32)): # 这里使用的是批量batch的处理方式，可以提高效率
        batch_sens = questions[i:i+32]
        inputs = tokenizer(batch_sens, padding=True, return_tensors="pt", max_length=128, truncation=True)
        inputs = {k:v.to(model.device)for k,v in inputs.items()}
        question_vector = model.bert(**inputs)[1]
        questions_vectors.append(question_vector)
questions_vectors = torch.concat(questions_vectors, dim=0).cpu().numpy()
print(questions_vectors.shape)

# 4. 创建索引
index = faiss.IndexFlatIP(768)
faiss.normalize_L2(questions_vectors)
index.add(questions_vectors)
print(index)

# 5. 编码问题
question = "在法律中定金与订金的区别订金和定金哪个"
with torch.inference_mode():
    inputs = tokenizer(question, padding=True, return_tensors="pt", max_length=128, truncation=True)
    inputs = {k:v.to(model.device)for k,v in inputs.items()}
    question_vector = model.bert(**inputs)[1].cpu().numpy()
print(question_vector.shape)

# 6. 向量匹配
faiss.normalize_L2(question_vector)
scores, indexes = index.search(question_vector, 10)
topk_results = data.values[indexes[0].tolist()]
print(topk_results[:,0])

# 7. 交互模型
cross_model = BertForSequenceClassification.from_pretrained("../05_文本相似度/向量匹配策略解决大候选样本的解决方案/checkpoint-250")
cross_model = cross_model.cuda()
cross_model.eval()

# 最终预测
candidate = topk_results[:,0].tolist()
ques = [question] * len(candidate)
inputs = tokenizer(ques, candidate, padding=True, return_tensors="pt", max_length=128, truncation=True)
inputs = {k:v.to(cross_model.device)for k,v in inputs.items()}
with torch.inference_mode():
    logits = cross_model(**inputs).logits.squeeze()
    print(logits)
    result = torch.argmax(logits, dim=-1)
print(result)

candidate_answers = topk_results[:,1].tolist()
match_question = candidate[result.item()]
final_answer = candidate_answers[result.item()]
print(f"问题：{match_question}\n回答：{final_answer}")
