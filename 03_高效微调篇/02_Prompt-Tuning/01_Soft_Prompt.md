# SoftPrompt

## 配置PEFT

```python
config = PromptTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=10)
model = get_peft_model(model, config)
print(model.print_trainable_parameters())
```

- 配置 Prompt Tuning 参数：
  - `task_type=TaskType.CAUSAL_LM`：任务类型为因果语言模型。
  - `num_virtual_tokens=10`：使用 10 个虚拟 token 作为 prompt。
- 使用 `get_peft_model` 将基础模型包装为 PEFT 模型。
- 打印可训练参数的数量。

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
      (embedding): Embedding(10, 2048)
    )
  )
)
'''
```

可以看到模型新加了一个`Prompt Embedding`参数：`   (embedding): Embedding(10, 2048)`   

