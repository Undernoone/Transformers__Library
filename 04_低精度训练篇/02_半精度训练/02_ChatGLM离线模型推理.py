import torch
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
adapter_path = "./02_ChatGLM_半精度训练/checkpoint-750"
model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True, torch_dtype=torch.bfloat16)
print(model)
print(model.config.torch_dtype)
model = PeftModel.from_pretrained(model, adapter_path)
print(model)
print(model.config.torch_dtype)
model.eval()

# 进行推理
ipt = tokenizer("Human: {}\n{}".format("人工智能是什么？", "").strip() + "\n\nAssistant: ", return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**ipt, max_length=512, do_sample=True, eos_token_id=tokenizer.eos_token_id)[0], skip_special_tokens=True))