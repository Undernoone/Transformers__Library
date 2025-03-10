# BitFit微调

## 只训练模型中带bias的参数

```python
for name, param in model.named_parameters():
    if "bias" not in name:
        param.requires_grad = False
    else:
        num_param += param.numel()
```