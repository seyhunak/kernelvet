# Legacy build kept for users without CMake
NVCC ?= nvcc
CUDA_HOME ?= /usr/local/cuda
ARCH := -arch=sm_80

all: test_vector_add gemm

test_vector_add: tests/test_vector_add.cu src/vector_add.cu
	$(NVCC) $(ARCH) -Iinclude -o $@ $^

gemm: src/gemm.cpp
	$(NVCC) $(ARCH) -Iinclude -o $@ $^ -L$(CUDA_HOME)/lib64 -lcudart -lcublas
