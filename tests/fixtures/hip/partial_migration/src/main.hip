#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <cublas_v2.h>
#include <cstdio>

__constant__ float scale;

__global__ void scale_kernel(float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int lane = threadIdx.x & 31;
  float v = __shfl_xor_sync(0xffffffff, x[i], 1);
  if (i < n && lane == 0) x[i] = v * scale;
}

int main() {
  float h = 2.0f;
  hipMemcpyToSymbol(scale, &h, sizeof(float));
  float* d;
  hipMalloc(&d, 1024 * sizeof(float));
  hipLaunchKernelGGL(scale_kernel, dim3(4), dim3(256), 0, 0, d, 1024);
  cudaDeviceSynchronize();
  cublasHandle_t handle;
  cublasCreate(&handle);
  hipFree(d);
  return 0;
}
