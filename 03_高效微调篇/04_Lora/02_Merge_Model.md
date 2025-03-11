# 合并模型

1. **导入库**：
   - `PeftModel`：用于加载和应用Peft微调模型。
   - `AutoModelForCausalLM` 和 `AutoTokenizer`：用于加载预训练的语言模型和对应的分词器。
2. **加载基础模型和分词器**：
   - `model = AutoModelForCausalLM.from_pretrained("Langboat/bloom-1b4-zh")`：加载名为 `Langboat/bloom-1b4-zh` 的预训练语言模型。
   - `tokenizer = AutoTokenizer.from_pretrained("Langboat/bloom-1b4-zh")`：加载与该模型对应的分词器。
   - `print("基础模型：",model)`：打印基础模型的信息。
3. **使用基础模型生成文本**：
   - `ipt = tokenizer("Human: {}\n{}".format("考试有哪些技巧？", "").strip() + "\n\nAssistant: ", return_tensors="pt")`：将输入文本编码为模型可接受的格式。
   - `print(tokenizer.decode(model.generate(**ipt, min_length=20, max_length=40, do_sample=False)[0], skip_special_tokens=True))`：使用基础模型生成文本并解码输出。
4. **加载Peft模型**：
   - `peft_model = PeftModel.from_pretrained(model, model_id="./Lora/checkpoint-125/")`：加载在基础模型上应用了Peft微调的模型，微调模型的路径为 `./Lora/checkpoint-125/`。
   - `print("Peft模型：",peft_model)`：打印Peft模型的信息。
5. **使用Peft模型生成文本**：
   - `ipt = tokenizer("Human: {}\n{}".format("考试有哪些技巧？", "").strip() + "\n\nAssistant: ", return_tensors="pt")`：再次编码输入文本。
   - `print(tokenizer.decode(peft_model.generate(**ipt, min_length=20, max_length=40, do_sample=False)[0], skip_special_tokens=True))`：使用Peft模型生成文本并解码输出。
6. **合并模型**：
   - `merge_model = peft_model.merge_and_unload()`：将Peft微调后的模型与基础模型合并，生成一个新的合并模型。
   - `print(merge_model)`：打印合并后的模型信息。
7. **使用合并模型生成文本**：
   - `ipt = tokenizer("Human: {}\n{}".format("考试有哪些技巧？", "").strip() + "\n\nAssistant: ", return_tensors="pt")`：再次编码输入文本。
   - `print(tokenizer.decode(merge_model.generate(**ipt, min_length=20, max_length=40, do_sample=False)[0], skip_special_tokens=True))`：使用合并后的模型生成文本并解码输出。
8. **保存合并模型**：
   - `merge_model.save_pretrained("./Lora/merge_model")`：将合并后的模型保存到指定路径 `./Lora/merge_model`。

### 总结：

这段代码展示了如何加载一个预训练的基础模型，应用Peft技术进行微调，并将微调后的模型与基础模型合并保存。整个过程包括模型的加载、微调、合并以及生成文本的示例。