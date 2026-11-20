#include "common.cuh"

__global__ void vector_add_kernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

void vector_add(const float* a, const float* b, float* c, int n, cudaStream_t stream) {
  float *da, *db, *dc;
  size_t bytes = n * sizeof(float);
  CUDA_CHECK(cudaMalloc(&da, bytes));
  CUDA_CHECK(cudaMalloc(&db, bytes));
  CUDA_CHECK(cudaMalloc(&dc, bytes));
  CUDA_CHECK(cudaMemcpyAsync(da, a, bytes, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(db, b, bytes, cudaMemcpyHostToDevice, stream));
  vector_add_kernel<<<ceil_div(n, 256), 256, 0, stream>>>(da, db, dc, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpyAsync(c, dc, bytes, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);
}
