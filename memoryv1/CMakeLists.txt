# 


cmake_minimum_required(VERSION 3.10)
project(chat_faiss)

set(CMAKE_CXX_STANDARD 17)

# 找 CMake 自带的 Python3 组件
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# 包含目录 - FAISS 头文件的位置
include_directories(/usr/local/include ${Python3_INCLUDE_DIRS})

# 库目录 - FAISS 库文件的位置
link_directories(/usr/local/lib)

# 源文件
add_executable(chat_faiss main.cpp)

# 链接 FAISS、OpenBLAS、Python3 库
target_link_libraries(chat_faiss
    faiss
    openblas
    ${Python3_LIBRARIES}
)
