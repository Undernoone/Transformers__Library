"""
Author: Coder729
Date: 2025/3/10
Description: 
"""
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Langboat/bloom-1b4-zh")
tokenizer = AutoTokenizer.from_pretrained("Langboat/bloom-1b4-zh")

p_model = PeftModel.from_pretrained(model, model_id="./Lora/checkpoint-1/")
print(p_model)

ipt = tokenizer("Human: {}\n{}".format("考试有哪些技巧？", "").strip() + "\n\nAssistant: ", return_tensors="pt")
tokenizer.decode(p_model.generate(**ipt, do_sample=False)[0], skip_special_tokens=True)

merge_model = p_model.merge_and_unload()
print(merge_model)

ipt = tokenizer("Human: {}\n{}".format("考试有哪些技巧？", "").strip() + "\n\nAssistant: ", return_tensors="pt")
tokenizer.decode(merge_model.generate(**ipt, do_sample=False)[0], skip_special_tokens=True)

merge_model.save_pretrained("./Lora/merge_model")