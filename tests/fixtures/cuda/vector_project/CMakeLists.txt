cmake_minimum_required(VERSION 3.20)
project(vector_project LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 80 86)

add_library(kernels src/vector_add.cu src/reduce.cu)
target_include_directories(kernels PUBLIC include)

add_executable(gemm src/gemm.cpp)
target_link_libraries(gemm PRIVATE kernels CUDA::cudart CUDA::cublas)

enable_testing()
add_executable(test_vector_add tests/test_vector_add.cu)
target_link_libraries(test_vector_add PRIVATE kernels)
add_test(NAME vector_add COMMAND test_vector_add)
