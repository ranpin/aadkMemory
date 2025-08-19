import os

# 设置 HF 镜像（Qwen 从 HuggingFace 镜像加载）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset
import json

from http import HTTPStatus
import dashscope
# dashscope.api_key = "your-api-key"
dashscope.api_key = "sk-2d0a7a049bde460cae67c64babf06d83"

# ========= Embedding 模型 =========
print("[Embedding] 加载模型中...")
# model = SentenceTransformer('shibing624/text2vec-base-chinese')
model = SentenceTransformer('thenlper/gte-base-zh')
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

    # """加载 SQuAD 问答数据集"""
    # try:
    #     from datasets import load_dataset
    #     # print("[HF] 加载 SQuAD 中文问答数据集...")
    #     # dataset = load_dataset("squad_v2", split=f"train[:{n}]")
    #     # texts = [f"A: {item['question']} B: {item['answers']['text'][0]}" for item in dataset]
    #     # return texts

    # except Exception as e:
    #     print(f"[HF][错误] 加载失败: {e}")
    #     print("[HF] 回退到内置测试数据")

    file_path = os.path.join(os.path.dirname(__file__), "./datasets/historyv1.json")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON格式错误: 不是数组")
        return data[:n]  # 返回前n条
    except Exception as e:
        print(f"[Python] 读取 history.json 发生错误: {e}")
        return []


# ========= Qwen 模型 =========
print("[Qwen] 加载模型中...")
device = "cuda" if torch.cuda.is_available() else "cpu"


# # QWEN_MODEL = "Qwen/Qwen-1_8B-Chat"
# # QWEN_MODEL = "Qwen/Qwen3-4B"
# # QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct-1M"
# QWEN_MODEL = "Qwen/Qwen3-1.7B"

# tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
# qwen_model = AutoModelForCausalLM.from_pretrained(
#     QWEN_MODEL,
#     torch_dtype=torch.float16 if device == "cuda" else torch.float32,
#     trust_remote_code=True
# ).to(device).eval()
# print("[Qwen] 模型加载完成.")


# def qwen_generate(prompt: str, max_new_tokens=256):
#     """调用 Qwen 模型生成回复"""
#     try:
#         inputs = tokenizer(prompt, return_tensors="pt").to(device)
#         if inputs["input_ids"].numel() == 0:
#             return "[Qwen] 输入无效"

#         # 强制禁用 past_key_values 以避免状态管理问题
#         outputs = qwen_model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             do_sample=True,
#             top_p=0.6,
#             temperature=0.6,
#             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id, # 填充标记ID
#             use_cache=False  # ★★★ 关键修复点 ★★★
#         )

#         input_len = inputs["input_ids"].shape[1]
#         gen_ids = outputs[0][input_len:]
#         reply = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
#         # reply = tokenizer.decode(gen_ids, skip_special_tokens=False).strip()  # 调试用，保留特殊标记

#         if not reply.strip():
#             return "[Qwen] 模型未生成有效文本"

#         return reply

#     except Exception as e:
#         return f"[Qwen] 异常: {str(e)}"




# ========= Qwen 云端调用 =========
DASHSCOPE_MODEL = "qwen2.5-14b-instruct"  # 云端模型名，可改成 qwen-plus/qwen-max

def qwen_generate(prompt: str, max_new_tokens=256):
    try:
        rsp = dashscope.Generation.call(
            model=DASHSCOPE_MODEL,
            prompt=prompt,
            max_tokens=max_new_tokens,
            temperature=0.6,
            top_p=0.6
        )

        # print(rsp)
        if rsp.status_code == HTTPStatus.OK:
            # API 会返回一个列表，你可以按文档取 output['text'] 或 output_text
            return rsp.output['text']
        else:
            return f"[Qwen API] 请求失败: {rsp.code}, {rsp.message}"

    except Exception as e:
        return f"[Qwen API 异常] {str(e)}"
    

# ========== 读取 car.json 并解析成 C++ 可用结构 ==========
def load_and_process_carjson(filepath: str):
    import os
    import json

    if not os.path.isabs(filepath):
        filepath = os.path.join(os.path.dirname(__file__), filepath)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except Exception as e:
        print(f"[Python] 无法读取 {filepath}:", e)
        return []

    # 内部解析函数
    def process_conversation(conversation_data):
        processed = []
        speaker_a = conversation_data["speaker_a"]
        speaker_b = conversation_data["speaker_b"]
        session_keys = [k for k in conversation_data.keys() if k.startswith("session_") and not k.endswith("_date_time")]
        for session_key in session_keys:
            timestamp_key = f"{session_key}_date_time"
            timestamp = conversation_data.get(timestamp_key, "")
            for dialog in conversation_data[session_key]:
                speaker = dialog["speaker"]
                text = dialog["text"]
                if "blip_caption" in dialog and dialog["blip_caption"]:
                    text += f" (image description: {dialog['blip_caption']})"
                if speaker == speaker_a:
                    processed.append({"user_input": text, "agent_response": "", "timestamp": timestamp})
                else:
                    if processed:
                        processed[-1]["agent_response"] = text
                    else:
                        processed.append({"user_input": "", "agent_response": text, "timestamp": timestamp})
        return processed

    all_data = []

    # 支持两种格式：数组形式 or 单对象形式
    if isinstance(dataset, list):
        for sample in dataset:
            conv = sample.get("conversation", {})
            proc = process_conversation(conv)
            all_data.extend(proc)

    elif isinstance(dataset, dict):
        conv = dataset.get("conversation", {})
        proc = process_conversation(conv)
        all_data.extend(proc)

    return all_data
