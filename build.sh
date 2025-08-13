#!/bin/bash
g++ studydemo/faiss_demo.cpp \
    -I/home/vision/chenrunbin/faiss \
    -o faiss_demo \
    -std=c++11 \
    -L/home/vision/chenrunbin/faiss \
    -lfaiss \
    -fopenmp \
    -lopenblas
