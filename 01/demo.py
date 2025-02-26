"""
Author: Coder729
Date: 2025/2/26
Description: 
"""
# 导入gradio
import gradio as gr
# 导入transformers相关包
from transformers import pipeline
# 通过Interface加载pipeline并启动阅读理解服务
# 如果无法通过这种方式加载，可以采用离线加载的方式
qa_pipeline = pipeline("question-answering", model="uer/roberta-base-chinese-extractive-qa", device=0)  # 0 表示 GPU

gr.Interface.from_pipeline(qa_pipeline).launch()
