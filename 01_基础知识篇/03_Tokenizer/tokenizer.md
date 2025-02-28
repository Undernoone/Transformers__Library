# Transformers tokenizer库

这段代码主要展示了如何使用`transformers`库中的`AutoTokenizer`进行中文文本的分词、编码和解码。下面是逐行的详细解析和专业见解：

### **定义输入句子**

```python
sentence = "我正在学习自然语言处理。" 
```

- **功能说明**：定义了一个中文句子，作为分词器的输入。这个句子将用于演示分词和编码的过程。

###  **加载预训练分词器**

```python
tokenizers = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")
```

- **功能说明**：从`transformers`库加载一个名为`uer/roberta-base-finetuned-dianping-chinese`的预训练模型分词器。这是一个针对中文进行微调的RoBERTa模型的分词器。通过`from_pretrained`方法，加载了模型的分词器，并将其赋值给`tokenizers`对象。

###  **分词**

```python
tokens = tokenizers.tokenize(sentence) 
```

- **功能说明**：使用加载的分词器对输入句子进行分词。`tokenize`方法将句子切分为一个个token（词或子词）。返回的结果是一个token列表。

### **打印分词结果**

```python
print("tokens:",tokens)
```

- **功能说明**：输出分词后的结果。`tokens`变量包含了句子中每个词的token。

### **将tokens转换为id**

```python
ids = tokenizers.convert_tokens_to_ids(tokens)
```

- **功能说明**：将分词后的tokens转换为对应的ID。每个token在预训练模型的词汇表中都有一个唯一的整数ID，`convert_tokens_to_ids`方法就是实现这个转换的功能。

### **打印ID结果**

```python
print("ids:",ids)
```

- **功能说明**：输出将tokens转换为ids后的结果，ID列表表示文本在模型中的表示。

### **直接编码文本**

```python
ids_encode = tokenizers.encode(sentence)
```

- **功能说明**：`encode`方法直接将句子转换为ID列表。它不仅会包括输入文本的tokens对应的ID，还会添加特殊符号，如`[CLS]`和`[SEP]`，这些符号在BERT系列模型中用于标识文本的开始和结束。

### **打印编码后的ID**

```python
print("ids_encode:",ids_encode)
```

- **功能说明**：输出直接编码后的ID列表，包含特殊符号。

### **ID到tokens的转换**

```python
print("tokens_encode:",tokenizers.convert_ids_to_tokens(ids))
```

- **功能说明**：将ID列表转换回tokens。`convert_ids_to_tokens`方法是将模型ID映射回相应的token，可以帮助我们查看ID对应的原始词汇。

### **ID到文本的解码**

```python
print("ids_decode:",tokenizers.decode(ids_encode))
```

- **功能说明**：将编码后的ID列表解码为文本。`decode`方法将ID转换回原始的字符串，同时处理特殊符号。

### **限制最大长度的分词**

```python
print("限制最大长度:",tokenizers.tokenize(sentence,max_length=3))
```

- **功能说明**：使用`tokenize`方法时，传入`max_length`参数，限制分词结果的最大长度为3。这会截断句子，保证返回的token数量不超过3个。

### **直接调用tokenizer**

```python
print("直接调用tokenizer:",tokenizers(sentence))
```

- **功能说明**：直接传入句子到分词器中，返回更多的信息，如token对应的ID、attention mask、token type IDs等。`attention mask`用于指示哪些token是实际的文本，哪些是填充符（padding）。

### **批量分词**

```python
sentences = ["我正在学习自然语言处理。", "你好，世界！","自然语言处理是一门很有趣的学科。"]
print("批量分词:",tokenizers(sentences))
```

- **功能说明**：将多个句子传入分词器进行批量处理。返回的结果是多个句子的token信息。

