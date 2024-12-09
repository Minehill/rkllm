cmake_minimum_required(VERSION 3.10)
project(rkllm_demo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 Arrow 和 Parquet 库
find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)

# 手动指定 lz4、zstd、re2、protobuf 和 thrift 的位置
find_path(LZ4_INCLUDE_DIR NAMES lz4.h)
find_library(LZ4_LIBRARY NAMES lz4)

if (LZ4_INCLUDE_DIR AND LZ4_LIBRARY)
    set(LZ4_FOUND TRUE)
else()
    set(LZ4_FOUND FALSE)
endif()

if (NOT LZ4_FOUND)
    message(FATAL_ERROR "Could not find lz4 library")
endif()

find_path(ZSTD_INCLUDE_DIR NAMES zstd.h)
find_library(ZSTD_LIBRARY NAMES zstd)

if (ZSTD_INCLUDE_DIR AND ZSTD_LIBRARY)
    set(ZSTD_FOUND TRUE)
else()
    set(ZSTD_FOUND FALSE)
endif()

if (NOT ZSTD_FOUND)
    message(FATAL_ERROR "Could not find zstd library")
endif()

find_path(RE2_INCLUDE_DIR NAMES re2/re2.h)
find_library(RE2_LIBRARY NAMES re2)

if (RE2_INCLUDE_DIR AND RE2_LIBRARY)
    set(RE2_FOUND TRUE)
else()
    set(RE2_FOUND FALSE)
endif()

if (NOT RE2_FOUND)
    message(FATAL_ERROR "Could not find re2 library")
endif()

find_path(Protobuf_INCLUDE_DIR NAMES google/protobuf/message.h)
find_library(Protobuf_LIBRARY NAMES protobuf)

if (Protobuf_INCLUDE_DIR AND Protobuf_LIBRARY)
    set(Protobuf_FOUND TRUE)
else()
    set(Protobuf_FOUND FALSE)
endif()

if (NOT Protobuf_FOUND)
    message(FATAL_ERROR "Could not find protobuf library")
endif()

find_path(Thrift_INCLUDE_DIR NAMES thrift/Thrift.h)
find_library(Thrift_LIBRARY NAMES thrift)

if (Thrift_INCLUDE_DIR AND Thrift_LIBRARY)
    set(Thrift_FOUND TRUE)
else()
    set(Thrift_FOUND FALSE)
endif()

if (NOT Thrift_FOUND)
    message(FATAL_ERROR "Could not find thrift library")
endif()

# 包含头文件路径
include_directories(${ARROW_INCLUDE_DIRS})
include_directories(${PARQUET_INCLUDE_DIRS})
include_directories(${LZ4_INCLUDE_DIR})
include_directories(${ZSTD_INCLUDE_DIR})
include_directories(${RE2_INCLUDE_DIR})
include_directories(${Protobuf_INCLUDE_DIR})
include_directories(${Thrift_INCLUDE_DIR})

set(SOURCE_FILES_1 src/test.cpp)
add_executable(llm_demo ${SOURCE_FILES_1})

# 链接 Arrow、Parquet、lz4、zstd、re2、protobuf 和 Thrift 库
target_link_libraries(llm_demo ${ARROW_LIBRARIES} ${PARQUET_LIBRARIES}
        ${LZ4_LIBRARY} ${ZSTD_LIBRARY} ${RE2_LIBRARY} ${Protobuf_LIBRARY} ${Thrift_LIBRARY})

set(SOURCE_FILES_2 src/multimodel_demo.cpp)
add_executable(multimodel_demo ${SOURCE_FILES_2})

set(RKLLM_API_PATH "${CMAKE_SOURCE_DIR}/../../runtime/${CMAKE_SYSTEM_NAME}/librkllm_api")
include_directories(${RKLLM_API_PATH}/include)
if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    set(RKLLM_RT_LIB ${RKLLM_API_PATH}/${CMAKE_ANDROID_ARCH_ABI}/librkllmrt.so)
    find_package(OpenMP REQUIRED)
    target_link_libraries(llm_demo  ${RKLLM_RT_LIB} log OpenMP::OpenMP_CXX)
    target_link_libraries(multimodel_demo  ${RKLLM_RT_LIB} log OpenMP::OpenMP_CXX)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(RKLLM_RT_LIB ${RKLLM_API_PATH}/aarch64/librkllmrt.so)
    target_link_libraries(llm_demo  ${RKLLM_RT_LIB} ${ARROW_LIBRARIES} ${PARQUET_LIBRARIES}
            ${LZ4_LIBRARY} ${ZSTD_LIBRARY} ${RE2_LIBRARY} ${Protobuf_LIBRARY} ${Thrift_LIBRARY})
    target_link_libraries(multimodel_demo  ${RKLLM_RT_LIB} ${ARROW_LIBRARIES} ${PARQUET_LIBRARIES}
            ${LZ4_LIBRARY} ${ZSTD_LIBRARY} ${RE2_LIBRARY} ${Protobuf_LIBRARY} ${Thrift_LIBRARY})
endif()