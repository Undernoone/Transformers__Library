# HardPrompt

## **配置 Hard Prompt Tuning**

```python
config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=len(tokenizer("下面是一段人与机器人的对话。")["input_ids"]),
    prompt_tuning_init=PromptTuningInit.TEXT,
    prompt_tuning_init_text="下面是一段人与机器人的对话。",
    tokenizer_name_or_path="Langboat/bloom-1b4-zh"
)
model = get_peft_model(model, config)
print(model.print_trainable_parameters())
```

- Hard Prompt Tuning 配置：
  - `prompt_tuning_init=PromptTuningInit.TEXT`：使用固定文本初始化 prompt。
  - `prompt_tuning_init_text="下面是一段人与机器人的对话。"`：指定 prompt 文本。
  - `num_virtual_tokens`：根据 prompt 文本的长度动态设置虚拟 token 数量。
- 使用 `get_peft_model` 将基础模型包装为 PEFT 模型，并打印可训练参数的数量。

## 模型对比

```txt
'''
可以看到经过Peft后基础模型外面加了一个PromptEmbedding
PeftModelForCausalLM(
  (base_model): BloomForCausalLM()
  (prompt_encoder): ModuleDict
  (
    (default): PromptEmbedding
    (
      (embedding): Embedding(8, 2048)
    )
  )
)
'''
```

可以看到模型新加了一个`Prompt Embedding`参数：`   (embedding): Embedding(8, 2048)`   

## 验证

```python
text = "下面是一段人与机器人的对话。"
tokenized_text = tokenizer(text, return_tensors="pt")
print(tokenized_text["input_ids"])
print(len(tokenized_text["input_ids"][0])) 
```

验证`"下面是一段人与机器人的对话。"`是不是8个token。