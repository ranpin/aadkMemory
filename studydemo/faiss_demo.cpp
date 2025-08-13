// faiss_demo.cpp
#include <faiss/IndexFlat.h>
#include <faiss/IndexIDMap.h>
#include <faiss/index_io.h>

#include <fstream>
#include <iostream>
#include <vector>
#include <string>

#include "json.hpp" // nlohmann/json

struct Item {
    int64_t id;
    std::vector<float> vec;
    std::string text;
};

// 读取 vectors.json
std::vector<Item> load_vectors(const std::string &path) {
    std::ifstream in(path);
    nlohmann::json j;
    in >> j;

    std::vector<Item> items;
    for (auto &e : j) {
        Item it;
        it.id = e["id"].get<int64_t>();
        it.text = e["text"].get<std::string>();
        it.vec = e["vector"].get<std::vector<float>>();
        items.push_back(std::move(it));
    }
    return items;
}

// 读取 query.json
std::vector<float> load_query(const std::string &path) {
    std::ifstream in(path);
    nlohmann::json j;
    in >> j;
    return j["vector"].get<std::vector<float>>();
}

int main() {
    // ====== 1. 加载向量数据 ======
    auto items = load_vectors("vectors.json");
    size_t dim = items[0].vec.size();
    std::cout << "维度=" << dim << ", 条数=" << items.size() << "\n";

    // ====== 2. 创建索引 ======
    faiss::IndexFlatIP base_index(dim); // 内积
    faiss::IndexIDMap2 id_index(&base_index); // 带ID索引
    for (auto &it : items) {
        id_index.add_with_ids(1, it.vec.data(), &it.id);
    }
    std::cout << "构建完成, 总条数=" << id_index.ntotal << "\n";

    // ====== 3. 加载查询 ======
    auto query_vec = load_query("query.json");
    int top_k = 3;
    std::vector<int64_t> ids(top_k);
    std::vector<float> sims(top_k);

    id_index.search(1, query_vec.data(), top_k, sims.data(), ids.data());

    std::cout << "查询结果:\n";
    for (int i = 0; i < top_k; i++) {
        std::cout << "Rank " << i+1 << ": id=" << ids[i]
                  << ", score=" << sims[i] << "\n";
    }

    // ====== 4. 删除例子 ======
    faiss::IDSelectorRange selector_del(2, 3); // 删除 id=2
    size_t removed = id_index.remove_ids(selector_del);
    std::cout << "删除id=2, 数量=" << removed
              << ", 剩余总条数=" << id_index.ntotal << "\n";

    // ====== 5. 更新例子 ======
    // 更新id=1的向量 -> 先删除再添加
    faiss::IDSelectorRange selector_upd(1, 2);
    id_index.remove_ids(selector_upd);
    Item upd = items[0];
    // 可以修改upd.vec内的数据，这里简单相同数据
    id_index.add_with_ids(1, upd.vec.data(), &upd.id);
    std::cout << "更新完成, 总条数=" << id_index.ntotal << "\n";

    // ====== 6. 保存索引 ======
    faiss::write_index(&id_index, "index.faiss");
    std::cout << "索引已保存到 index.faiss\n";

    // ====== 7. 重新加载索引 ======
    faiss::Index *loaded_index = faiss::read_index("index.faiss");
    std::cout << "重新加载后条数: " << loaded_index->ntotal << "\n";
    delete loaded_index;

    return 0;
}
