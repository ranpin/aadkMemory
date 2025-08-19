#include "MemoryManager.hpp"
#include <iostream>

ShortTermMemory::ShortTermMemory(int capacity, int dim)
    : capacity(capacity), dim(dim), base_index(dim), index(&base_index), next_id(1) {}

void ShortTermMemory::add(const QAData& qa, const std::vector<float>& embedding) {
    if (id_queue.size() >= capacity) {
        faiss::idx_t oldest = id_queue.front();
        id_queue.pop_front();
        faiss::IDSelectorRange sel(oldest, oldest+1);
        index.remove_ids(sel);
        id2qa.erase(oldest);
    }
    index.add_with_ids(1, embedding.data(), &next_id);
    id2qa[next_id] = qa;
    id_queue.push_back(next_id);
    next_id++;
}

std::vector<RetrievalResult> ShortTermMemory::retrieve(const std::vector<float>& query_vec, int k, float threshold) {
    std::vector<RetrievalResult> results;
    std::vector<faiss::idx_t> ids(k);
    std::vector<float> sims(k);
    index.search(1, query_vec.data(), k, sims.data(), ids.data());
    for (int i=0; i<k && sims[i] > threshold; i++) {
        auto it = id2qa.find(ids[i]);
        if (it != id2qa.end()) {
            RetrievalResult rr;
            rr.qa = it->second;
            rr.similarity = sims[i];
            results.push_back(rr);
        }
    }
    return results;
}

bool ShortTermMemory::isFull() const {
    return id_queue.size() >= capacity;
}

QAData ShortTermMemory::popOldest() {
    faiss::idx_t oldest = id_queue.front();
    id_queue.pop_front();
    QAData qa = id2qa[oldest];
    faiss::IDSelectorRange sel(oldest, oldest+1);
    index.remove_ids(sel);
    id2qa.erase(oldest);
    return qa;
}

void ShortTermMemory::printIndex() {
    int display_id = 1;
    for (auto faiss_id : id_queue) {
        auto &qa = id2qa[faiss_id];
        std::cout << "ID " << display_id << ": Q=" << qa.user_input << " A=" << qa.agent_response << "\n";
        display_id++;
    }
}
