import os

# 设置 HF 镜像（Qwen 从 HuggingFace 镜像加载）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset


# ========= Embedding 模型 =========
print("[Embedding] 加载模型中...")
model = SentenceTransformer('shibing624/text2vec-base-chinese')
print("[Embedding] 模型加载完成.")


def encode(text: str):
    """编码单句文本"""
    try:
        vec = model.encode(text, normalize_embeddings=True)
        return vec.tolist()
    except Exception as e:
        print("[Python][encode] 编码单句失败:", e)
        return []


def encode_list(texts: list, batch_size=16):
    """批量编码文本"""
    all_vecs = []
    try:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            vecs = model.encode(batch, normalize_embeddings=True)
            all_vecs.extend([v.tolist() for v in vecs])
            print(f"[Python][encode_list] 已编码 {i+len(batch)}/{len(texts)} 条")
    except Exception as e:
        print("[Python][encode_list] 批量编码失败:", e)
    return all_vecs


def load_lccc(n=50):
    # """
    # 从 Hugging Face 加载中文对话数据
    # 如果无法加载则回退到内置测试数据
    # """
    # try:
    #     print(f"[HF] 正在加载 Hugging Face 数据集 (前 {n} 条)...")
        
    #     # 示例：加载 Hugging Face 上的中文数据集（需替换为实际存在的数据集）
    #     # 本例使用 ChnSentiCorp 数据集作为演示
    #     dataset = load_dataset("lansinuote/ChnSentiCorp", split="train")
        
    #     texts = []
    #     for i, item in enumerate(dataset):
    #         if i >= n:
    #             break
    #         # 根据实际数据集字段调整
    #         text = f"{item.get('text', '')} {item.get('label', '')}".strip()
    #         if text:
    #             texts.append(text)
        
    #     print(f"[HF] 成功加载 {len(texts)} 条对话")
    #     return texts

    """加载 SQuAD 中文问答数据集"""
    try:
        from datasets import load_dataset
        print("[HF] 加载 SQuAD 中文问答数据集...")
        dataset = load_dataset("squad_v2", split=f"train[:{n}]")
        texts = [f"问题: {item['question']} 答案: {item['answers']['text'][0]}" for item in dataset]
        return texts

    except Exception as e:
        print(f"[HF][错误] 加载失败: {e}")
        print("[HF] 回退到内置测试数据")
        return [
            "你好，最近怎么样？ 很好，谢谢。",
            "今天天气不错啊 是的，非常适合散步。",
            "周末一起去看电影吧 好啊，一起去。",
            "你在干嘛 我在看书。",
            "吃饭了吗 还没吃呢。"
        ][:n]


# ========= Qwen 模型 =========
print("[Qwen] 加载模型中...")
device = "cuda" if torch.cuda.is_available() else "cpu"
QWEN_MODEL = "Qwen/Qwen-1_8B-Chat"

tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
qwen_model = AutoModelForCausalLM.from_pretrained(
    QWEN_MODEL,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    trust_remote_code=True
).to(device).eval()
print("[Qwen] 模型加载完成.")


def qwen_generate(prompt: str, max_new_tokens=256):
    """调用 Qwen 模型生成回复"""
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        if inputs["input_ids"].numel() == 0:
            return "[Qwen] 输入无效"

        # 强制禁用 past_key_values 以避免状态管理问题
        outputs = qwen_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            use_cache=False  # ★★★ 关键修复点 ★★★
        )

        if outputs is None or outputs.dim() == 0:
            return "[Qwen] 生成失败：无有效输出"

        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_len:]

        if generated_ids.numel() == 0:
            return "[Qwen] 生成失败：无新内容"

        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    except Exception as e:
        return f"[Qwen] 异常: {str(e)}"

