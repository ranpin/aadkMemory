#include <Python.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIDMap.h>
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <cctype> 
#include <numeric>
#include <faiss/Index.h>
#include <ctime>     // ★ 时间相关
#include <iomanip>   // ★ std::put_time
#include <sstream>   // ★ std::ostringstream
#include <deque>

// 工具函数: Python list -> C++ vector<float>
std::vector<float> py_list_to_vector(PyObject* pyList) {
    std::vector<float> emb;
    if (!PyList_Check(pyList)) return emb;
    Py_ssize_t size = PyList_Size(pyList);
    emb.resize(size);
    for (Py_ssize_t i = 0; i < size; i++) {
        emb[i] = static_cast<float>(PyFloat_AsDouble(PyList_GetItem(pyList, i)));
    }
    return emb;
}

// 调用 Python encode(text)
std::vector<float> get_embedding(PyObject* pModule, const std::string& text) {
    std::vector<float> emb;
    PyObject* pFunc = PyObject_GetAttrString(pModule, "encode");
    if (pFunc && PyCallable_Check(pFunc)) {
        PyObject* pArgs = PyTuple_Pack(1, PyUnicode_FromString(text.c_str()));
        PyObject* pValue = PyObject_CallObject(pFunc, pArgs);
        Py_DECREF(pArgs);
        if (pValue) {
            emb = py_list_to_vector(pValue);
            Py_DECREF(pValue);
        }
    }
    Py_XDECREF(pFunc);
    return emb;
}

// 调用 Python load_lccc(n)
std::vector<std::string> load_lccc_texts(PyObject* pModule, int n) {
    std::vector<std::string> texts;
    PyObject* pFunc = PyObject_GetAttrString(pModule, "load_lccc");
    if (pFunc && PyCallable_Check(pFunc)) {
        PyObject* pArgs = PyTuple_Pack(1, PyLong_FromLong(n));
        PyObject* pValue = PyObject_CallObject(pFunc, pArgs);
        Py_DECREF(pArgs);
        if (pValue && PyList_Check(pValue)) {
            Py_ssize_t size = PyList_Size(pValue);
            for (Py_ssize_t i = 0; i < size; i++) {
                texts.push_back(PyUnicode_AsUTF8(PyList_GetItem(pValue, i)));
            }
            Py_DECREF(pValue);
        }
    }
    Py_XDECREF(pFunc);
    return texts;
}

// 调用 Python encode_list
std::vector<std::vector<float>> encode_list(PyObject* pModule, const std::vector<std::string>& texts) {
    std::vector<std::vector<float>> vecs;
    PyObject* pFunc = PyObject_GetAttrString(pModule, "encode_list");
    if (pFunc && PyCallable_Check(pFunc)) {
        PyObject* pyList = PyList_New(texts.size());
        for (size_t i = 0; i < texts.size(); i++) {
            PyList_SetItem(pyList, i, PyUnicode_FromString(texts[i].c_str()));
        }
        PyObject* pArgs = PyTuple_Pack(1, pyList);
        PyObject* pValue = PyObject_CallObject(pFunc, pArgs);
        Py_DECREF(pArgs);
        if (!pValue) {
            PyErr_Print(); // 建议加上这一句
        }
        if (pValue && PyList_Check(pValue)) {
            Py_ssize_t outer = PyList_Size(pValue);
            vecs.resize(outer);
            for (Py_ssize_t i = 0; i < outer; i++) {
                vecs[i] = py_list_to_vector(PyList_GetItem(pValue, i));
            }
            Py_DECREF(pValue);
        }
    }
    Py_XDECREF(pFunc);
    return vecs;
}

// 调用 Python qwen_generate
std::string qwen_generate(PyObject* pModule, const std::string& prompt) {
    std::string reply;
    PyObject* pFunc = PyObject_GetAttrString(pModule, "qwen_generate");
    if (pFunc && PyCallable_Check(pFunc)) {
        PyObject* pArgs = PyTuple_Pack(1, PyUnicode_FromString(prompt.c_str()));
        PyObject* pValue = PyObject_CallObject(pFunc, pArgs);
        Py_DECREF(pArgs);
        if (!pValue) {
            PyErr_Print(); // 建议加上这一句
        }

        if (pValue) {
            reply = PyUnicode_AsUTF8(pValue);
            reply.erase(
                std::remove_if(reply.begin(), reply.end(),
                        [](unsigned char x){ return std::isspace(x); }),
                reply.end()
            );
            if (reply.empty()) reply = "[C++] 收到空回复";
            Py_DECREF(pValue);
        } else {
            PyErr_Print();
            reply = "[C++] Python 调用失败";
        }
    }
    Py_XDECREF(pFunc);
    return reply;
}


int main() {
    Py_Initialize();
    PyRun_SimpleString("import sys");
    // embed.py 搜索路径：优先用环境变量 AADK_EMBED_PATH（绝对路径）；否则同时加入
    // 当前目录与上级目录——从 memoryv1/ 或 memoryv1/build/ 运行都能命中。
    // （原先这里写死了作者机器的绝对路径，换机器必失败。）
    PyRun_SimpleString("import os");
    PyRun_SimpleString("_p = os.environ.get('AADK_EMBED_PATH')");
    PyRun_SimpleString("sys.path.append(_p) if _p else sys.path.extend(['.', '..'])");

    PyObject* pModule = PyImport_ImportModule("embed");
    if (!pModule) {
        PyErr_Print();
        std::cerr << "[C++] 无法导入 embed.py\n";
        return -1;
    }

    // 1. 加载数据
    int N_init = 50; // 少量调试
    auto texts = load_lccc_texts(pModule, N_init);
    if (texts.empty()) {
        std::cerr << "[C++] LCCC 数据加载失败\n";
        return -1;
    }

    // 2. 批量编码
    auto vecs = encode_list(pModule, texts);
    if (vecs.empty() || vecs[0].empty()) {
        std::cerr << "[C++] 批量编码失败，vecs 为空\n";
        return -1;
    }
    int dim = vecs[0].size();
    std::cout << "[C++] 返回向量数: " << vecs.size() << " 每个维度: " << dim << "\n";

    // 3. 构建 FAISS
    faiss::IndexFlatIP base_index(dim);
    faiss::IndexIDMap index(&base_index);
    std::map<faiss::idx_t, std::string> id2text;
    std::deque<faiss::idx_t> id_queue;
    faiss::idx_t next_id = 1;
    float sims_threshold = 0.4; // 相似度阈值

    for (size_t i = 0; i < vecs.size(); ++i) {
        if (vecs[i].empty()) {
            std::cerr << "[警告] 第 " << i << " 条为空，跳过\n";
            continue;
        }
        index.add_with_ids(1, vecs[i].data(), &next_id);
        id2text[next_id] = texts[i];
        id_queue.push_back(next_id);    // ✅初始化历史记录也加入队列
        next_id++;
    }
    std::cout << "[初始化完成] 条数: " << index.ntotal << "\n";
    //打印出初始的对话记录索引表
    std::cout << "[初始对话记录]:" << std::endl;
    for (const auto& pair : id2text) {
        std::cout << "ID: " << pair.first << " 文本: " << pair.second << std::endl;
    }
    std::cout << std::endl<< std::endl;


    // 4. 对话循环，进行5轮对话
    for (int i = 0; i < 5; i++) {
        std::string query;
        std::cout << "=========================================================================\n";
        std::cout << "Question: ";
        std::getline(std::cin, query);
        if (query == "exit" || query == "quit") break;

        auto qvec = get_embedding(pModule, query);
        if (qvec.empty()) {
            std::cerr << "[C++] 查询向量为空，跳过\n";
            continue;
        }

        int k = 5;
        std::vector<faiss::idx_t> ids(k);
        std::vector<float> sims(k);
        index.search(1, qvec.data(), k, sims.data(), ids.data());


        std::cout << "检索结果如下：\n相关历史片段（按相似度排序）：\n";
        for (int i = 0; i < k && sims[i] > sims_threshold; i++) {
            if (ids[i] != -1) {
                std::cout << "- [相似度: " << sims[i] << "] " << id2text[ids[i]] << "\n";
            }
        }
        std::cout << "\n\n";

        // 获取当前时间戳
        std::time_t now = std::time(nullptr);
        std::tm* tm_info = std::localtime(&now);
        std::ostringstream oss;
        oss << std::put_time(tm_info, "%Y-%m-%d %H:%M:%S");
        std::string time_str = oss.str();

        std::string prompt = "Question:" + query+ " \"timestamp\": "+ time_str + "\n";
        prompt += "相关可供参考的历史对话片段信息如下:\n";
        for (int i = 0; i < k && sims[i] > sims_threshold; i++) {
            if (ids[i] != -1) {
                prompt += id2text[ids[i]];
            }
        }
        prompt += "\n请参考上述信息，用中文回复用户的问题。";

        std::cout << "Qwen prompt: " << prompt << "\n\n";

        std::string reply = qwen_generate(pModule, prompt);
        std::cout << "Qwen answer: " << reply << "\n\n";


        // // 加入记忆v1（仅添加查询）
        // index.add_with_ids(1, qvec.data(), &next_id);
        // id2text[next_id] = query;
        // next_id++;


        // 加入记忆v2
        // index.add_with_ids(1, qvec.data(), &next_id);
        // id2text[next_id] = query;
        // next_id++;
        // auto emb_reply = get_embedding(pModule, reply);
        // if (!emb_reply.empty()) {
        //     index.add_with_ids(1, emb_reply.data(), &next_id);
        //     id2text[next_id] = reply;
        //     next_id++;
        // }

        // 加入记忆v3:将问答对和此刻的时间戳一块作为历史记忆，限制条数，先入先出，动态加入到索引表中
        std::string qa_pair = "Question: " + query + " Answer: " + reply + " \"timestamp\": "+ time_str;   

        // 设定最大容量（包含历史 + 新增）
        const size_t MAX_CAPACITY = 10;        //需要大于等于整个索引表的大小

        // 添加QA
        auto emb_qa = get_embedding(pModule, qa_pair);
        if (!emb_qa.empty() && emb_qa.size() == dim) {

            // 容量满 -> 移除最旧
            if (id_queue.size() >= MAX_CAPACITY) {
                faiss::idx_t oldest = id_queue.front();
                id_queue.pop_front();
                faiss::IDSelectorRange sel(oldest, oldest + 1);
                index.remove_ids(sel);
                id2text.erase(oldest);
            }

            // 添加新(真实 faiss ID)
            index.add_with_ids(1, emb_qa.data(), &next_id);
            id2text[next_id] = qa_pair;
            id_queue.push_back(next_id);
            next_id++;
        }

        std::cout << "当前对话已加入索引表，当前索引条数：" << id_queue.size() <<  std::endl;
        // 打印
        std::cout << "[Index内容]:" << std::endl;
        int display_id = 1;
        for (auto faiss_id : id_queue) {
            std::cout << "ID: " << display_id
                    << " 文本: " << id2text[faiss_id] << std::endl;
            display_id++;
        }


        std::cout << std::endl << std::endl;

    }

    Py_XDECREF(pModule);
    Py_Finalize();
    return 0;
}
