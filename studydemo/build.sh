#!/bin/bash
# FAISS 安装前缀：默认 /usr/local（studydemo/build_faiss.sh 的默认安装位置，
# 头文件在 $PREFIX/include/faiss、库在 $PREFIX/lib）。装在别处就用环境变量覆盖，
# 例如 FAISS_PREFIX=$HOME/faiss bash studydemo/build.sh
FAISS_PREFIX="${FAISS_PREFIX:-/usr/local}"
g++ studydemo/faiss_demo.cpp \
    -I"$FAISS_PREFIX/include" \
    -o faiss_demo \
    -std=c++11 \
    -L"$FAISS_PREFIX/lib" \
    -lfaiss \
    -fopenmp \
    -lopenblas
