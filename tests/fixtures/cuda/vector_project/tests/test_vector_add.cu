#include <cmath>
#include <vector>
#include "common.cuh"

int main() {
  const int n = 1 << 16;
  std::vector<float> a(n), b(n), c(n);
  for (int i = 0; i < n; ++i) { a[i] = i * 0.25f; b[i] = 1.0f; }
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));
  vector_add(a.data(), b.data(), c.data(), n, s);
  CUDA_CHECK(cudaStreamDestroy(s));
  for (int i = 0; i < n; ++i)
    if (std::fabs(c[i] - (a[i] + b[i])) > 1e-5f) { std::printf("FAIL at %d\n", i); return 1; }
  std::printf("PASS\n");
  return 0;
}
