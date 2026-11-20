#include "common.cuh"

__device__ __forceinline__ unsigned lane_id() {
  unsigned id;
  asm volatile("mov.u32 %0, %%laneid;" : "=r"(id));
  return id;
}

__device__ float warp_reduce_sum(float v) {
  for (int offset = 16; offset > 0; offset /= 2)
    v += __shfl_down_sync(0xffffffff, v, offset);
  return v;
}

__global__ void block_reduce_kernel(const float* in, float* out, int n) {
  __shared__ float partial[32];
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float v = tid < n ? in[tid] : 0.0f;
  v = warp_reduce_sum(v);
  int lane = threadIdx.x % 32;
  int warp = threadIdx.x / 32;
  if (lane == 0) partial[warp] = v;
  __syncthreads();
  if (warp == 0) {
    v = lane < blockDim.x / WARP_SIZE ? partial[lane] : 0.0f;
    v = warp_reduce_sum(v);
    if (lane == 0) atomicAdd(out, v);
  }
#if __CUDA_ARCH__ >= 700
  __syncwarp();
#endif
}

float block_reduce_sum(const float* data, int n) {
  float *din, *dout, result = 0.0f;
  CUDA_CHECK(cudaMalloc(&din, n * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&dout, sizeof(float)));
  *dout = 0.0f;
  CUDA_CHECK(cudaMemcpy(din, data, n * sizeof(float), cudaMemcpyHostToDevice));
  block_reduce_kernel<<<ceil_div(n, 1024), 1024>>>(din, dout, n);
  CUDA_CHECK(cudaDeviceSynchronize());
  result = *dout;
  cudaFree(din);
  cudaFree(dout);
  return result;
}
