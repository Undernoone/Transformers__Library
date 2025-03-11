"""
Author: Coder729
Date: 2025/3/11
Description: 
"""
import torch
from torch import nn
from peft import *


model = nn.Sequential(
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 2)
)

print(model)

for name, module in model.named_modules():
    print(name)

config = LoraConfig(target_modules=["0"])

lora_model = get_peft_model(model, config)
print(lora_model)

model2