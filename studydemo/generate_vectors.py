import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# generate_vectors.py


from sentence_transformers import SentenceTransformer
import json

model = SentenceTransformer('shibing624/text2vec-base-chinese')

# 样本文本
texts = [
    "你好，世界",
    "这是一个测试",
    "向量检索很有趣",
    "Python和C++可以配合使用"
]

data = []
for idx, text in enumerate(texts):
    emb = model.encode(text, normalize_embeddings=True).tolist()
    data.append({
        "id": idx + 1,
        "text": text,
        "vector": emb
    })

with open("vectors.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False)
print("已保存 vectors.json")

# 准备一个查询向量（查询文本由用户输入）
query_text = input("请输入查询文本：")
query_emb = model.encode(query_text, normalize_embeddings=True).tolist()

with open("query.json", "w", encoding="utf-8") as f:
    json.dump({"text": query_text, "vector": query_emb}, f, ensure_ascii=False)
print("已保存 query.json")
