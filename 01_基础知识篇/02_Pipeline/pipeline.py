"""
Author: Coder729
Date: 2025/2/26
Description: transformers pipeline
"""
import torch
from PIL.ImageDraw import ImageDraw
from transformers.pipelines import SUPPORTED_TASKS
from pprint import pprint
from transformers import pipeline, QuestionAnsweringPipeline
from datasets import load_dataset, Audio
from transformers import pipeline
from PIL import Image,ImageDraw

pprint(list(SUPPORTED_TASKS.keys()), width=100) # 查看Pipeline支持的任务类型

for k, v in SUPPORTED_TASKS.items(): # 查看Pipeline支持的任务类型对应的模型名称
    print(f"{k}: {v}", end="\n\n")

# 文本分类任务
pipe_for_text_classification = pipeline("text-classification", device=0) # 使用默认model:DistilBertForSequenceClassification
print(pipe_for_text_classification("I am a good person."), "\n", pipe_for_text_classification("I am a bad person."))

model = "uer/roberta-base-finetuned-dianping-chinese" # 如果不指定model则会使用默认model:DistilBertForSequenceClassification
tokenizer = "uer/roberta-base-finetuned-dianping-chinese" # 如果不指定tokenizer则会使用默认tokenizer:RobertaTokenizer
pipe_for_text_classification_model = pipeline("text-classification", model, tokenizer, device=0)
print(pipe_for_text_classification_model("I am a good person."), "\n", pipe_for_text_classification_model("I am a bad person."))

# 问答任务
pipe_for_question_answering = pipeline("question-answering", model="uer/roberta-base-chinese-extractive-qa", device=0)
print(pipe_for_question_answering(question="中国的首都是哪里？", context="中国的首都是北京", max_answer_len=3))

# 目标检测

model = "google/owlvit-base-patch32"
pipe_for_object_detection = pipeline("zero-shot-object-detection", model=model, device=0)
image_path = "image/object_detection.jpg"
image = Image.open(image_path)
predictions = pipe_for_object_detection(image, candidate_labels=["hat", "sunglasses", "book"], multi_label=True)
print(predictions)

if predictions:
    draw = ImageDraw.Draw(image)  # 创建绘图对象
    for prediction in predictions:
        box = prediction['box']  # 获取检测框的位置
        label = prediction['label']  # 获取标签
        score = prediction['score']  # 获取分数
        xmin, ymin, xmax, ymax = box['xmin'], box['ymin'], box['xmax'], box['ymax']
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
        draw.text((xmin, ymin), f"{label}: {score:.2f}", fill="red")
image.show()

# 语音识别
speech_recognizer = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h", device=0)

dataset = load_dataset("PolyAI/minds14", name="en-US", split="train", trust_remote_code=True)

dataset = dataset.cast_column("audio", Audio(sampling_rate=speech_recognizer.feature_extractor.sampling_rate))

audio_samples = dataset[:4]["audio"]
result = speech_recognizer(audio_samples)


