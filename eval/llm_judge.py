import json
import re
import os
from collections import defaultdict
import numpy as np
from openai import OpenAI
from typing import List, Dict

# 初始化
client = OpenAI(
        api_key='sk-4c4668e46a534650b187cb0ef63d8d24',
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1')

# 提示词模板（复用 Mem0 的 ACCURACY_PROMPT）
ACCURACY_PROMPT = """
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user), 
    (2) a ’original’ (ground truth) answer, 
    (3) a system generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The original answer will usually be a concise and short answer that includes the referenced topic. 
The system generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the original answer, it should be counted as CORRECT. 

For time related questions, the orginal answer will be a specific date, month, year, etc. 
The system answer might use relative time references (like "last Tuesday" or "next month"), 
but you should be generous with your grading - as long as it refers to the same date or time period, 
it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Original answer: {original_answer}
System answer: {system_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. 
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""

def extract_json(text):
    """
    Extracts JSON content from a string, removing enclosing triple backticks and optional 'json' tag if present.
    If no code block is found, returns the text as-is.
    """
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        json_str = text  # assume it's raw JSON
    return json_str

def evaluate_llm_judge(question: str, original_answer: str, system_answer: str) -> int:
    """使用 Qwen-Max 判断生成答案是否正确"""
    try:
        # 构造提示词
        prompt = ACCURACY_PROMPT.format(
            question=question,
            original_answer=original_answer,
            system_answer=system_answer
        )
        
        # 调用 Qwen-Max
        response = client.chat.completions.create(
            model="qwen-max",  # 使用 Qwen-Max
            messages=[
                {
                    "role": "user",
                    "content": ACCURACY_PROMPT.format(
                        question=question,
                        original_answer=original_answer,
                        system_answer=system_answer
                    ),
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.0,  # 确保输出确定性
        )
        
        # 提取 JSON 中的标签
        label = json.loads(extract_json(response.choices[0].message.content))["label"]
        return 1 if label == "CORRECT" else 0
    except Exception as e:
        print(f"Error in LLM Judge: {e}")
        return 0  # 出错时默认标记为 WRONG

def main(file_path: str = "all_loco_results.json"):
    """主函数：基于 LLM-as-judge 的评估"""
    # 加载数据
    with open(file_path, "r") as f:
        data = json.load(f)

    # 初始化结果存储
    LLM_JUDGE = defaultdict(list)
    RESULTS = []

    # 遍历所有样本
    for idx, sample in enumerate(data):
        category = sample["category"]
        if int(category) == 5:  # 跳过 category5
            continue

        question = sample["question"]
        original_answer = sample["original_answer"]
        system_answer = sample["system_answer"]

        # 调用 LLM 判定
        label = evaluate_llm_judge(question, original_answer, system_answer)

        # 保存结果
        RESULTS.append({
            "index": idx,
            "question": question,
            "original_answer": original_answer,
            "response": system_answer,
            "category": category,
            "llm_label": label
        })

        # 每处理 10 个样本保存一次中间结果
        if (idx + 1) % 10 == 0:
            output_path = "llm_judge_results.json"
            with open(output_path, "w") as f:
                json.dump(RESULTS, f, indent=4)
            print(f"已处理 {idx+1} 个样本，中间结果保存至 {output_path}")

        # 统计各类别准确率
        LLM_JUDGE[category].append(label)
        print("All categories accuracy:")
        for cat, results in LLM_JUDGE.items():
            acc = np.mean(results)
            print(f"  Category {cat}: {acc:.4f} ({sum(results)}/{len(results)})")
        print("------------------------------------------")

    # 保存最终结果
    output_path = "llm_judge_results.json"
    with open(output_path, "w") as f:
        json.dump(RESULTS, f, indent=4)
    print(f"最终结果已保存到 {output_path}")

    # 打印最终总结
    print("\nFinal Summary:")
    for cat, results in LLM_JUDGE.items():
        print(f"Category {cat}: Accuracy = {np.mean(results):.4f} ({sum(results)}/{len(results)})")

if __name__ == "__main__":
    main()
