cmake_minimum_required(VERSION 3.10)

# 项目名称
project(RKLLMExample)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 查找 Arrow 和 Parquet 库
find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)

# 添加可执行文件
add_executable(rkllm_api_demo src/test.cpp)

# 链接 Arrow 和 Parquet 库
# target_link_libraries(rkllm_api_demo PRIVATE Arrow::arrow_shared ${ARROW_LIBRARIES} ${PARQUET_LIBRARIES})
target_link_libraries(rkllm_api_demo PRIVATE Arrow::arrow_shared ${ARROW_LIBRARIES} ${PARQUET_LIBRARIES} /usr/lib/librkllmrt.so)

# 包含头文件目录
target_include_directories(rkllm_api_demo PRIVATE ${ARROW_INCLUDE_DIR} ${PARQUET_INCLUDE_DIR})
