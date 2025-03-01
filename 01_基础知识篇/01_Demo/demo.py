"""
Author: Coder729
Date: 2025/2/26
Description: 测试
"""
import gradio as gr # 交互Web界面
from transformers import pipeline
qa_pipeline = pipeline("question-answering", model="uer/roberta-base-chinese-extractive-qa", device=0)
gr.Interface.from_pipeline(qa_pipeline).launch()
