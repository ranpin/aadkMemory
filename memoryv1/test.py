import os

# 设置 HF 镜像（Qwen 从 HuggingFace 镜像加载）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
QWEN_MODEL = "Qwen/Qwen-1_8B-Chat"

tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    QWEN_MODEL,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    trust_remote_code=True
).to(device).eval()

prompt = "用户问题：你好，今天天气怎么样？具体是什么天气？"
# prompt += "- 问题: When did Beyonce take a hiatus in her career and take control of her management? 答案: 2010。"
prompt += "相关历史对话片段:A: 昨天天气好吗? B: 倾盆大雨。"
prompt += "相关历史对话片段:A: 今天天气好吗? B: 天气晴朗。"
prompt += "相关历史对话片段:A: 明天天气好吗? B: 漫天大雪。"
# prompt += "- 问题: Beyonce's childhood home believed in what religion? 答案: Methodist\n"
# prompt += "- 问题: When did Beyoncé release Dangerously in Love? 答案: 2003\n"
prompt += "请根据以上信息，用一句简洁的中文回复用户。"

inputs = tokenizer(prompt, return_tensors="pt").to(device)
print("input_ids:", inputs["input_ids"])
print("pad_token_id:", tokenizer.pad_token_id, tokenizer.eos_token_id)

outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    use_cache=False  # 避免 past_key_values None 报错
)

# print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print(tokenizer.decode(outputs[0], skip_special_tokens=False))

