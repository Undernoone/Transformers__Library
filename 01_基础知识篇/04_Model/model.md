# Tramsformers AutoModel

### 代码结构与解释：

1. **导入必要的库**

   ```python
   from transformers import AutoModel, AutoTokenizer, AutoConfig
   from transformers import BertModel, BertTokenizer, BertConfig
   from pprint import pprint
   from transformers import AutoModelForSequenceClassification
   ```

   - `AutoModel` 是 Hugging Face 提供的一个自动加载模型的工具类，可以根据指定的模型名称加载相应的预训练模型。
   - `AutoTokenizer` 用于加载与模型配套的分词器（tokenizer）。
   - `AutoConfig` 用于加载模型的配置文件，可以查看或修改模型的配置。
   - `pprint` 用于更好地格式化输出（特别是对于嵌套字典或列表）。
   - `AutoModelForSequenceClassification` 是针对序列分类任务的模型类（比如文本分类），但在此代码中没有使用。

2. **加载模型和配置**

   ```python
   model = AutoModel.from_pretrained("rbt3")
   print("Model config:", model.config)
   ```

   - `AutoModel.from_pretrained("rbt3")` 会自动加载 `rbt3` 预训练模型。如果没有下载，它会自动从 Hugging Face 的模型库中下载。
   - `model.config` 打印出模型的配置信息，如模型的架构（Transformer 类型、层数、隐藏层维度等）。

3. **加载模型的配置**

   ```python
   config = AutoConfig.from_pretrained("rbt3")
   ```

   - `AutoConfig.from_pretrained("rbt3")` 加载与模型相关的配置。这是一个配置对象，可以修改模型的参数配置（例如：修改隐藏层维度、调整学习率等）。

4. **进行文本编码和推理**

   ```python
   sentence = "I am studying NLP."
   
   tokenizer = AutoTokenizer.from_pretrained("rbt3")
   inputs = tokenizer(sentence, return_tensors="pt")
   pprint(inputs, width=100)
   outputs = model(**inputs)
   pprint(outputs, width=100)
   ```

   - 这里定义了一个待处理的文本 `sentence = "I am studying NLP."`。
   - 使用 `AutoTokenizer.from_pretrained("rbt3")` 加载分词器，`tokenizer(sentence, return_tensors="pt")` 会将文本转换为模型可以处理的张量格式（`pt` 表示返回 PyTorch 张量）。返回的 `inputs` 包含了编码后的输入，通常是一个字典，包含了如 `input_ids`, `attention_mask` 等字段。
   - `model(**inputs)` 将编码后的输入传递给模型进行推理，`outputs` 包含了模型的输出，通常是 `logits` 或其他可能的信息（如隐藏状态等）。

5. **加载带有注意力机制的模型**

   ```python
   model_with_attentions = AutoModel.from_pretrained("rbt3", output_attentions=True)
   ```

   - `output_attentions=True` 会让模型返回注意力权重（attention weights）。这对于分析模型的注意力机制非常有用，通常我们可以用来可视化模型的注意力集中在哪些部分。

6. **再次进行推理并查看注意力信息**

   ```python
   outputs = model_with_attentions(**inputs)
   pprint(outputs, width=100)
   ```

   - 这里再次使用带有注意力输出的模型进行推理，`outputs` 不仅包含模型的输出（例如 `logits`），还可能包含注意力权重。如果你需要查看注意力权重，`outputs.attentions` 就是你需要的部分。