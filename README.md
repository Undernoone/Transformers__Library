# Transformers 库的学习笔记

Transformers库的学习笔记。

原视频地址：[Transformers](https://space.bilibili.com/21060026)

原作者：https://github.com/zyds

这个仓库包含了与 Hugging Face Transformers 库相关的学习笔记、代码片段、教程和示例，在原作者的基础上进行了部分修改，更加细致，更适合初学者。

## 代码结构

```text
root
│  01_基础知识篇
└─ │  01
   │  02_Pipline
   │  ...
   └─ xx 
      │  xx.py
      │  xx.md # 由ChatGPT生成的代码解释，Prompt：https://bailian.console.aliyun.com/#/prompt-manage
│  02_实战演练篇
│  03_高效微调篇
│  README.md
│  Image
│  ...
│  xx
```

## 目录

### 01_基础知识篇

| 组件名称  | 笔记                                                         | 官方文档                                                     | 编辑时间  |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ | --------- |
| Pipeline  | [Pipelline](01_基础知识篇/02_Pipeline/pipeline.md)           | [Pipeline](https://huggingface.co/docs/transformers/main/en/quicktour#pipeline) | 2025/2/26 |
| Tokenizer | [Tokenizer](01_基础知识篇/03_Tokenizer/tokenizer.md)         | [Tokenizer](https://huggingface.co/docs/transformers/main/en/quicktour#autotokenizer) | 2025/2/26 |
| Model     | [Model](01_基础知识篇/04_Model/model.md)   [Model实战](01_基础知识篇/04_Model/实战.md) | [AutoModel](https://huggingface.co/docs/transformers/main/en/quicktour#automodel) | 2025/2/27 |
| Datasets  | [Datasets](01_基础知识篇/05_Datasets/datasets.md)            |                                                              |           |
| Evaluate  | [Evaluate](01_基础知识篇/06_Evaluate/evaluate.md)            |                                                              | 2025/2/27 |
| Trainer   | [Trainer](01_基础知识篇/07_Trainer/trainer.md)               |                                                              | 2025/2/28 |

## 02_实战演练篇

| 组件名称         | 笔记                                                         | 官方文档                                                     | 编辑时间  |
| ---------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | --------- |
| 如何降低显存占用 | [降低显存占用](02_实战演练篇/01_如何降低显存占用/降低显存占用实战.md) | [Methods and tools for efficient training on a single GPU](https://huggingface.co/docs/transformers/perf_train_gpu_one#methods-and-tools-for-efficient-training-on-a-single-gpu) | 2025/2/28 |
| 命名实体识别     | [命名实体识别](02_实战演练篇/02_命名实体识别/命名实体识别.md) |                                                              | 2025/3/01 |
| 机器阅读理解     | [截断策略](02_实战演练篇/03_机器阅读理解/截断策略机器阅读理解)  [滑动窗口（待完成）](02_实战演练篇/03_机器阅读理解/滑动窗口机器阅读理解) |                                                              | 2025/3/02 |
| 多项选择         | [多项选择](02_实战演练篇/04_多项选择/多项选择.md)            |                                                              | 2025/3/03 |
| 文本相似度       | [分类方法](02_实战演练篇/05_文本相似度/01_分类方法计算文本相似度)  [回归方法](02_实战演练篇/05_文本相似度/02_回归方法计算文本相似度)  [回归优化（待完成）](02_实战演练篇/05_文本相似度/03_回归方法过大时解决方案) |                                                              | 2025/3/04 |
| 检索对话机器人   | [检索对话机器人](02_实战演练篇/06_检索对话机器人/对话机器人.md) |                                                              | 2025/3/05 |
| 预训练模型       | [掩码语言模型](02_实战演练篇/07_预训练模型/01_掩码语言模型.md)  [因果语言模型](02_实战演练篇/07_预训练模型/02_因果语言模型.md) |                                                              | 2025/3/07 |
| 文本摘要         | [T5模型](02_实战演练篇/08_文本摘要/T5模型文本摘要.md)  [GLM模型（待完成）](02_实战演练篇/08_文本摘要/GLM模型文本摘要.md) |                                                              | 2025/3/08 |
| 对话机器人       | [对话机器人](02_实战演练篇/09_对话机器人/对话机器人.md)      |                                                              | 2025/3/08 |

## 扩展知识

| 组件名称                        | 笔记                                                         | 编辑时间  |
| ------------------------------- | ------------------------------------------------------------ | --------- |
| 评价指标                        | [评价指标](扩展知识/查看不同分类任务的评价指标.md)           | 2025/2/28 |
| 修改Huggingface文件默认下载地址 | [修改Huggingface默认存储地址](扩展知识/修改Huggingface文件默认下载地址) | 2025/3/01 |
| Checkpointing                   | [Checkpointing](扩展知识/Checkpointing.md)                   | 2025/3/01 |

