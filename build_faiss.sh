#!/bin/bash
set -e  # 出错就退出

# ===== 配置 =====
# FAISS 源码路径（就是这个脚本所在目录）
SRC_DIR=$(cd $(dirname $0) && pwd)
BUILD_DIR=${SRC_DIR}/build

echo "源码目录: $SRC_DIR"
echo "构建目录: $BUILD_DIR"

# ===== 清理旧构建 =====
if [ -d "${BUILD_DIR}" ]; then
    echo "删除旧的 build 目录..."
    rm -rf "${BUILD_DIR}"
fi

# ===== CMake 配置 =====
echo "开始 CMake 配置..."
cmake -B "${BUILD_DIR}" \
    -DFAISS_ENABLE_GPU=OFF \            # 关闭 GPU 支持
    -DFAISS_ENABLE_PYTHON=OFF \         # 关闭 Python 绑定
    -DCMAKE_BUILD_TYPE=Release \        # 优化编译
    -DBUILD_SHARED_LIBS=ON \            # 生成动态库
    -DFAISS_OPT_LEVEL=generic \         # ARM 通用优化
    -DBLA_VENDOR=OpenBLAS \             # 显式使用 OpenBLAS
    -DOpenMP_C_FLAGS="-fopenmp" \       # 开启 OpenMP
    -DOpenMP_CXX_FLAGS="-fopenmp"       # C++ 端 OpenMP

# ===== 编译 =====
echo "开始编译..."
cmake --build "${BUILD_DIR}" -j"$(nproc)"

# ===== 安装 =====
echo "开始安装 (需要 sudo 权限)..."
sudo cmake --install "${BUILD_DIR}"

echo "✅ FAISS 编译 & 安装完成"
echo "    头文件: /usr/local/include/faiss"
echo "    库文件: /usr/local/lib/libfaiss.so"
