# aadkMemory


# Description

aadkMemory is a library for memory allocation and deallocation.




# Installation
## 安装python环境
conda create -n aadkMemory python=3.10
conda activate aadkMemory
缺啥模块就装啥模块


## 安装FAISS C++版
参考：https://github.com/facebookresearch/faiss/blob/main/INSTALL.md


# Usage
项目结构：

编译：
mkdir build && cd build
cmake ..
make -j8

运行：
./chat_faiss


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