cmake_minimum_required(VERSION 3.10)
project(rkllm_demo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(nlohmann_json 3.2.0 REQUIRED)

add_executable(llm_demo src/readjson.cpp)

target_link_libraries(llm_demo nlohmann_json::nlohmann_json)

set(RKLLM_API_PATH "${CMAKE_SOURCE_DIR}/../../runtime/${CMAKE_SYSTEM_NAME}/librkllm_api")
include_directories(${RKLLM_API_PATH}/include)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(RKLLM_RT_LIB ${RKLLM_API_PATH}/aarch64/librkllmrt.so)
    target_link_libraries(llm_demo  ${RKLLM_RT_LIB})

endif()

