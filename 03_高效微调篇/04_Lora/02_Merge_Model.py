"""
Author: Coder729
Date: 2025/3/10
Description: 
"""
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
# 基础模型
model = AutoModelForCausalLM.from_pretrained("Langboat/bloom-1b4-zh")
tokenizer = AutoTokenizer.from_pretrained("Langboat/bloom-1b4-zh")
print("基础模型：",model)

# Peft模型
peft_model = PeftModel.from_pretrained(model, model_id="./Lora/checkpoint-125/")
print("Peft模型：",peft_model)

ipt = tokenizer("Human: {}\n{}".format("考试有哪些技巧？", "").strip() + "\n\nAssistant: ", return_tensors="pt")
print(tokenizer.decode(peft_model.generate(**ipt, do_sample=False)[0], skip_special_tokens=True))

# 合并模型
merge_model = peft_model.merge_and_unload()
print(merge_model)

ipt = tokenizer("Human: {}\n{}".format("考试有哪些技巧？", "").strip() + "\n\nAssistant: ", return_tensors="pt")
print(tokenizer.decode(merge_model.generate(**ipt, do_sample=False)[0], skip_special_tokens=True))

merge_model.save_pretrained("./Lora/merge_model")