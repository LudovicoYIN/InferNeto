cmake_minimum_required(VERSION 3.16)
project(InferNeto)
set(CMAKE_CXX_STANDARD 17)
# 添加编译器标志
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -mfma")
find_package(benchmark REQUIRED)
find_package(OpenMP REQUIRED)
find_package(glog REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(GTest REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
set(link_lib glog::glog GTest::gtest)
if(!WIN32)
    set(link_lib "${link_lib} pthread")
endif()
set(link_math_lib  ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
aux_source_directory(./test/tensor TEST_TENSOR)
aux_source_directory(./test/graph TEST_GRAPH)
aux_source_directory(./test/model TEST_MODEL)
aux_source_directory(./core/data/cpu CPU_TENSOR_SOURCE)
aux_source_directory(./core/infer/pnnx PNNX_SOURCE)
aux_source_directory(./core/infer/ INFER_SOURCE)
aux_source_directory(./core/node/abstract NODE_ABSTRACT_SOURCE)
aux_source_directory(./core/node/details NODE_DETAILS_SOURCE)
aux_source_directory(./core/node/parser PARSER_SOURCE)
add_executable(InferNeto main.cpp ${TEST_TENSOR} ${TEST_GRAPH} ${TEST_MODEL} ${CPU_TENSOR_SOURCE} ${PNNX_SOURCE} ${INFER_SOURCE} ${NODE_ABSTRACT_SOURCE} ${NODE_DETAILS_SOURCE} ${PARSER_SOURCE})

target_link_libraries(InferNeto ${link_lib} ${OpenCV_LIBS} ${link_math_lib} OpenMP::OpenMP_CXX)

target_include_directories(InferNeto PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(InferNeto PUBLIC ${GTest_INCLUDE_DIR})
target_include_directories(InferNeto PUBLIC ./core)
enable_testing()