// MemoryManager.hpp
#pragma once
#include <faiss/IndexFlat.h>
#include <faiss/IndexIDMap.h>
#include <string>
#include <vector>
#include <deque>
#include <map>

struct QAData {
    std::string user_input;     // 用户输入
    std::string agent_response;     // 代理回复
    std::string timestamp;      // 时间戳
};

struct RetrievalResult {
    QAData qa;  // 对话数据
    float similarity;   // 相似度
};

class ShortTermMemory {
public:
    ShortTermMemory(int capacity, int dim);     // 构造函数
    void add(const QAData& qa, const std::vector<float>& embedding);    // 添加对话记录
    std::vector<RetrievalResult> retrieve(const std::vector<float>& query_vec, int k, float threshold);     // 检索对话记录
    bool isFull() const;    // 是否已满
    QAData popOldest();     // 移除最旧的对话记录
    void printIndex();      // 打印索引

private:
    int capacity;   // 容量
    int dim;    // 向量维度
    std::deque<faiss::idx_t> id_queue;  // ID 队列
    std::map<faiss::idx_t, QAData> id2qa;   // ID 到 QAData 的映射
    faiss::IndexFlatIP base_index;      // 基础索引
    faiss::IndexIDMap index;    // ID 映射索引
    faiss::idx_t next_id;   // 下一个 ID
};

class MidTermMemory {
public:
    void add(const QAData& qa) {}
    std::vector<QAData> retrieve(const std::string& query) { return {}; }
};

class LongTermMemory {
public:
    void updateUserProfile(const std::string& new_profile) {}
    std::vector<std::string> retrieveKnowledge(const std::string& query) { return {}; }
};
