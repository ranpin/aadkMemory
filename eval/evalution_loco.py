import json
import re
from typing import List, Dict
from collections import defaultdict
import statistics

def simple_tokenize(text: str) -> List[str]:
    """Simple tokenization function."""
    if not text:
        return []
    
    # Convert to string if not already
    text = str(text).lower()
    # Remove punctuation and split by whitespace using regex (正确的方法)
    tokens = re.findall(r'\b\w+\b', text)
    return tokens

def calculate_f1(prediction: str, reference: str) -> float:
    """Calculate F1 score for prediction against reference."""
    # Tokenize both prediction and reference
    pred_tokens = set(simple_tokenize(prediction))
    ref_tokens = set(simple_tokenize(reference))
    
    # Calculate intersection
    common_tokens = pred_tokens & ref_tokens
    
    # Calculate precision and recall
    precision = len(common_tokens) / len(pred_tokens) if len(pred_tokens) > 0 else 0
    recall = len(common_tokens) / len(ref_tokens) if len(ref_tokens) > 0 else 0
    
    # Calculate F1 score
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0
    return f1

def load_data(file_path: str) -> List[Dict]:
    """Load data from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def main(file_path: str):
    """Main function to calculate average F1 scores per category."""
    # Load data from file
    data = load_data(file_path)
    
    # Initialize category dictionary
    category_f1 = defaultdict(list)

    # 新增：存储低 F1 样本
    low_f1_samples = []  # 存储所有 F1 < 0.5 的样本
    threshold = 1.1     # 低 F1 分数阈值（可调）
    
    
    # Calculate F1 scores for each sample
    for sample in data:
        category = sample['category']
        if int(category)==5:
            continue
        system_answer = sample['system_answer']
        original_answer = sample['original_answer']
        
        # Calculate F1 score
        f1 = calculate_f1(system_answer, original_answer)

        # 新增：保存低 F1 样本
        if f1 < threshold:
            low_f1_samples.append({
                "category": category,
                "question": sample.get("question", "N/A"),  # 从数据中提取问题
                "original_answer": original_answer,
                "system_answer": system_answer,
                "f1": f1
            })
        
        # Append F1 score to the corresponding category
        category_f1[category].append(f1)
    
    # 新增：保存低 F1 样本到文件
    with open("low_f1_samples.json", "w", encoding="utf-8") as f:
        json.dump(low_f1_samples, f, indent=4)
    print(f"低 F1 样本已保存到 low_f1_samples.json(共 {len(low_f1_samples)} 个)")
    

    # Calculate and print average F1 scores for each category
    for category, f1_scores in category_f1.items():
        avg_f1 = statistics.mean(f1_scores)
        print(f"Category {category}: Average F1 Score = {avg_f1:.4f}")

if __name__ == "__main__":
    file_path = "all_loco_results.json"  # 使用main_loco_parse.py生成的文件
    main(file_path)