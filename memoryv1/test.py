from embed import qwen_generate, tokenizer, qwen_model

# 测试空输入
print(qwen_generate(""))  # 应返回 "[Qwen] 输入无效"

# 测试有效输入
prompt = "用户问题：你好\n相关历史片段：\n- 你好，最近怎么样？ 很好，谢谢。\n基于以上内容，给出简洁自然的中文回复："
print(qwen_generate(prompt))  # 应返回有效回复

# 验证输入输出张量
inputs = tokenizer("测试", return_tensors="pt")
outputs = qwen_model.generate(**inputs, max_new_tokens=10)
print(outputs.shape)  # 应为 torch.Size([1, N])
