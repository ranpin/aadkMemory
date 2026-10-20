# aadkMemory


# Description

aadkMemory 是一个基于 FAISS 的对话记忆检索 demo，不是内存分配库。

工作流程：C++ 主程序（`memoryv1/main.cpp`）内嵌 Python 解释器调用 `memoryv1/embed.py`，
用 sentence-transformers（`thenlper/gte-base-zh`）把 `datasets/historyv1.json` 里的历史对话
向量化后建 FAISS 索引（IndexFlatIP + IndexIDMap）；进入交互循环后，每输入一个问题就做一次
相似度检索，把 Top-K 历史片段拼进 prompt，再调用 Qwen（dashscope 云端 `qwen2.5-14b-instruct`）
生成回复，并把本轮问答重新编码后写回索引。

**定位与边界（如实说明）**：这是一个学习性质的对话记忆检索 demo——embedding 在本地
（sentence-transformers）、生成走云端（dashscope Qwen），属于「本地检索 + 云端生成」的
功能分工，**不是端侧部署**，也没有端云融合路由或资源调度。

- `memoryv1/`：主程序，编译产物为 `chat_faiss`
- `studydemo/`：FAISS 独立学习示例（`faiss_demo.cpp`）与 FAISS 源码编译脚本




# Installation
## 安装python环境
conda create -n aadkMemory python=3.10
conda activate aadkMemory
缺啥模块就装啥模块

`memoryv1/embed.py` 实际用到的包：

```bash
pip install sentence-transformers transformers torch datasets dashscope
```

## 安装FAISS C++版
参考：https://github.com/facebookresearch/faiss/blob/main/INSTALL.md

`memoryv1/CMakeLists.txt` 除了 FAISS，还要求：
- OpenBLAS（链接 `-lopenblas`）
- Python3 开发头文件与库（`find_package(Python3 COMPONENTS Interpreter Development REQUIRED)`，因为主程序内嵌 Python）

也可以直接用 `studydemo/build_faiss.sh` 从源码编译并安装 FAISS 到 `/usr/local`。


# Usage
项目结构：

编译（CMakeLists.txt 在 `memoryv1/` 下，必须先进入该目录）：
cd memoryv1
mkdir build && cd build
cmake ..
make -j8

运行：
./chat_faiss

`memoryv1/main.cpp` 通过 `sys.path` 找 `embed.py`：默认同时搜索当前目录与上级目录，
所以从 `memoryv1/` 或 `memoryv1/build/` 运行都能命中。若 `embed.py` 放在别处，
用环境变量指定它所在目录的绝对路径：

```bash
AADK_EMBED_PATH=/abs/path/to/memoryv1 ./chat_faiss
```

`studydemo/build.sh` 的 FAISS 前缀默认 `/usr/local`（即 `build_faiss.sh` 的安装位置）；
装在别处用 `FAISS_PREFIX=<你的前缀> bash studydemo/build.sh` 覆盖。

调用 Qwen 走 dashscope 云端，需要先设置 API Key（`embed.py` 里的 `dashscope.api_key` 当前是注释掉的占位）。


# Example


# References

```
aadkMemory
├─ README.md
├─ memoryv1
│  ├─ CMakeLists.txt
│  ├─ __pycache__
│  │  └─ embed.cpython-310.pyc
│  ├─ build
│  │  ├─ CMakeCache.txt
│  │  ├─ CMakeFiles
│  │  │  ├─ 3.29.6
│  │  │  │  ├─ CMakeCCompiler.cmake
│  │  │  │  ├─ CMakeCXXCompiler.cmake
│  │  │  │  ├─ CMakeDetermineCompilerABI_C.bin
│  │  │  │  ├─ CMakeDetermineCompilerABI_CXX.bin
│  │  │  │  ├─ CMakeSystem.cmake
│  │  │  │  ├─ CompilerIdC
│  │  │  │  │  ├─ CMakeCCompilerId.c
│  │  │  │  │  ├─ a.out
│  │  │  │  │  └─ tmp
│  │  │  │  └─ CompilerIdCXX
│  │  │  │     ├─ CMakeCXXCompilerId.cpp
│  │  │  │     ├─ a.out
│  │  │  │     └─ tmp
│  │  │  ├─ CMakeConfigureLog.yaml
│  │  │  ├─ CMakeDirectoryInformation.cmake
│  │  │  ├─ CMakeScratch
│  │  │  ├─ Makefile.cmake
│  │  │  ├─ Makefile2
│  │  │  ├─ TargetDirectories.txt
│  │  │  ├─ chat_faiss.dir
│  │  │  │  ├─ DependInfo.cmake
│  │  │  │  ├─ build.make
│  │  │  │  ├─ cmake_clean.cmake
│  │  │  │  ├─ compiler_depend.internal
│  │  │  │  ├─ compiler_depend.make
│  │  │  │  ├─ compiler_depend.ts
│  │  │  │  ├─ depend.make
│  │  │  │  ├─ flags.make
│  │  │  │  ├─ link.txt
│  │  │  │  ├─ main.cpp.o
│  │  │  │  ├─ main.cpp.o.d
│  │  │  │  └─ progress.make
│  │  │  ├─ cmake.check_cache
│  │  │  ├─ pkgRedirects
│  │  │  └─ progress.marks
│  │  ├─ Makefile
│  │  ├─ chat_faiss
│  │  └─ cmake_install.cmake
│  ├─ datasets
│  │  ├─ historyv1.json
│  │  └─ historyv2.json
│  ├─ embed.py
│  ├─ main.cpp
│  ├─ test.py
│  └─ train.json
└─ studydemo
   ├─ build.sh
   ├─ build_faiss.sh
   ├─ faiss_demo
   ├─ faiss_demo.cpp
   ├─ generate_vectors.py
   ├─ index.faiss
   ├─ json.hpp
   ├─ query.json
   └─ vectors.json

```