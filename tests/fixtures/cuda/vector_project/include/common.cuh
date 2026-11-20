#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t err__ = (call);                                                 \
    if (err__ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n",                          \
                   cudaGetErrorString(err__), __FILE__, __LINE__);              \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

#define WARP_SIZE 32

__host__ __device__ inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

inline bool has_fast_fp64() {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  return prop.major >= 6;  // Pascal and newer
}

void vector_add(const float* a, const float* b, float* c, int n, cudaStream_t stream);
float block_reduce_sum(const float* data, int n);
