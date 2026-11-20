#include <cstdio>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <nvToolsExt.h>

int main() {
  const int n = 256;
  std::vector<float> a(n * n, 1.0f), b(n * n, 2.0f), c(n * n, 0.0f);
  float *da, *db, *dc;
  cudaMalloc(&da, a.size() * sizeof(float));
  cudaMalloc(&db, b.size() * sizeof(float));
  cudaMalloc(&dc, c.size() * sizeof(float));
  cudaMemcpy(da, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(db, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice);

  cublasHandle_t handle;
  cublasCreate(&handle);
  const float alpha = 1.0f, beta = 0.0f;
  nvtxRangePushA("sgemm");
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, da, n, db, n, &beta, dc, n);
  nvtxRangePop();
  cudaMemcpy(c.data(), dc, c.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cublasDestroy(handle);
  std::printf("c[0] = %f (expected %f)\n", c[0], 2.0f * n);
  cudaFree(da); cudaFree(db); cudaFree(dc);
  return c[0] == 2.0f * n ? 0 : 1;
}
