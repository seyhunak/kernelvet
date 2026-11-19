"""Benchmark workload sources. Every number comes from running these on the device."""

HIP_BENCH_SOURCE = r"""
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#include <algorithm>

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
  std::printf("ERROR %s %s\n", #x, hipGetErrorString(e)); return 1; } } while (0)

__global__ void vadd(const float* a, const float* b, float* c, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

template <int T>
__global__ void sgemm_tiled(const float* A, const float* B, float* C, int n) {
  __shared__ float As[T][T], Bs[T][T];
  int r = blockIdx.y * T + threadIdx.y, c = blockIdx.x * T + threadIdx.x;
  float acc = 0.f;
  for (int k0 = 0; k0 < n; k0 += T) {
    As[threadIdx.y][threadIdx.x] = (r < n && k0 + threadIdx.x < n) ? A[r * n + k0 + threadIdx.x] : 0.f;
    Bs[threadIdx.y][threadIdx.x] = (c < n && k0 + threadIdx.y < n) ? B[(k0 + threadIdx.y) * n + c] : 0.f;
    __syncthreads();
    for (int k = 0; k < T; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    __syncthreads();
  }
  if (r < n && c < n) C[r * n + c] = acc;
}

static float median(std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main(int argc, char** argv) {
  int iters = argc > 1 ? std::atoi(argv[1]) : 20;
  hipDeviceProp_t p; CHECK(hipGetDeviceProperties(&p, 0));
  std::printf("DEVICE %s|%s\n", p.gcnArchName, p.name);
  hipEvent_t s, e; CHECK(hipEventCreate(&s)); CHECK(hipEventCreate(&e));

  // launch latency: empty-ish kernel on 1 element
  {
    float *d; CHECK(hipMalloc(&d, 3 * sizeof(float)));
    std::vector<float> t;
    for (int i = 0; i < iters * 5; ++i) {
      CHECK(hipEventRecord(s)); vadd<<<1, 1>>>(d, d + 1, d + 2, 1); CHECK(hipEventRecord(e));
      CHECK(hipEventSynchronize(e)); float ms; CHECK(hipEventElapsedTime(&ms, s, e)); t.push_back(ms);
    }
    std::printf("RESULT kernel_launch_latency us %f %d\n", median(t) * 1000.f, iters * 5);
    hipFree(d);
  }
  // host<->device bandwidth (pinned)
  {
    size_t bytes = 256ull << 20; float *h, *d;
    CHECK(hipHostMalloc(&h, bytes)); CHECK(hipMalloc(&d, bytes));
    std::vector<float> th, td;
    for (int i = 0; i < iters; ++i) {
      CHECK(hipEventRecord(s)); CHECK(hipMemcpy(d, h, bytes, hipMemcpyHostToDevice)); CHECK(hipEventRecord(e));
      CHECK(hipEventSynchronize(e)); float ms; CHECK(hipEventElapsedTime(&ms, s, e)); th.push_back(ms);
      CHECK(hipEventRecord(s)); CHECK(hipMemcpy(h, d, bytes, hipMemcpyDeviceToHost)); CHECK(hipEventRecord(e));
      CHECK(hipEventSynchronize(e)); CHECK(hipEventElapsedTime(&ms, s, e)); td.push_back(ms);
    }
    std::printf("RESULT h2d_bandwidth GB/s %f %d\n", bytes / (median(th) * 1e6), iters);
    std::printf("RESULT d2h_bandwidth GB/s %f %d\n", bytes / (median(td) * 1e6), iters);
    hipHostFree(h); hipFree(d);
  }
  // allocation latency
  {
    std::vector<float> t;
    for (int i = 0; i < iters; ++i) {
      void* d; auto t0 = std::chrono::high_resolution_clock::now();
      CHECK(hipMalloc(&d, 64ull << 20)); CHECK(hipFree(d));
      auto t1 = std::chrono::high_resolution_clock::now();
      t.push_back(std::chrono::duration<float, std::micro>(t1 - t0).count());
    }
    std::printf("RESULT malloc_free_64MiB us %f %d\n", median(t), iters);
  }
  // vector add effective bandwidth
  {
    size_t n = 64ull << 20; float *a, *b, *c;
    CHECK(hipMalloc(&a, n * 4)); CHECK(hipMalloc(&b, n * 4)); CHECK(hipMalloc(&c, n * 4));
    CHECK(hipMemset(a, 0, n * 4)); CHECK(hipMemset(b, 0, n * 4));
    std::vector<float> t;
    for (int i = 0; i < iters; ++i) {
      CHECK(hipEventRecord(s)); vadd<<<(n + 255) / 256, 256>>>(a, b, c, n); CHECK(hipEventRecord(e));
      CHECK(hipEventSynchronize(e)); float ms; CHECK(hipEventElapsedTime(&ms, s, e)); t.push_back(ms);
    }
    std::printf("RESULT vector_add_bandwidth GB/s %f %d\n", 3.0 * n * 4 / (median(t) * 1e6), iters);
    hipFree(a); hipFree(b); hipFree(c);
  }
  // tiled sgemm
  {
    int n = 2048; size_t bytes = (size_t)n * n * 4; float *A, *B, *C;
    CHECK(hipMalloc(&A, bytes)); CHECK(hipMalloc(&B, bytes)); CHECK(hipMalloc(&C, bytes));
    CHECK(hipMemset(A, 0, bytes)); CHECK(hipMemset(B, 0, bytes));
    dim3 blk(16, 16), grd((n + 15) / 16, (n + 15) / 16);
    sgemm_tiled<16><<<grd, blk>>>(A, B, C, n); CHECK(hipDeviceSynchronize());
    std::vector<float> t;
    for (int i = 0; i < std::max(3, iters / 4); ++i) {
      CHECK(hipEventRecord(s)); sgemm_tiled<16><<<grd, blk>>>(A, B, C, n); CHECK(hipEventRecord(e));
      CHECK(hipEventSynchronize(e)); float ms; CHECK(hipEventElapsedTime(&ms, s, e)); t.push_back(ms);
    }
    std::printf("RESULT sgemm_2048_tiled GFLOP/s %f %d\n", 2.0 * n * n * (double)n / (median(t) * 1e6), (int)t.size());
    hipFree(A); hipFree(B); hipFree(C);
  }
  return 0;
}
"""

TORCH_BENCH_SOURCE = r"""
import json, sys, time
iters = int(sys.argv[1]) if len(sys.argv) > 1 else 20
inference = len(sys.argv) > 2 and sys.argv[2] == "1"
out = {"results": [], "device": None}
try:
    import torch
except Exception as exc:
    print(json.dumps({"error": f"{type(exc).__name__}: {exc}"})); sys.exit(0)
if not torch.cuda.is_available():
    print(json.dumps({"error": "torch.cuda.is_available() is False"})); sys.exit(0)
dev = torch.device("cuda")
out["device"] = torch.cuda.get_device_name(0)

def timed(fn, n):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    ts.sort(); return ts[len(ts) // 2]

for dtype, name in ((torch.float32, "fp32"), (torch.float16, "fp16")):
    n = 4096
    a = torch.randn(n, n, device=dev, dtype=dtype); b = torch.randn(n, n, device=dev, dtype=dtype)
    ms = timed(lambda: a @ b, iters)
    out["results"].append({"name": f"matmul_{n}_{name}", "unit": "TFLOP/s", "value": 2 * n**3 / (ms * 1e9), "iterations": iters})
x = torch.randn(64 * 1024 * 1024, device=dev)
ms = timed(lambda: x * 2.0 + 1.0, iters)
out["results"].append({"name": "elementwise_bandwidth", "unit": "GB/s", "value": 2 * x.numel() * 4 / (ms * 1e6), "iterations": iters})
h = torch.randn(64 * 1024 * 1024 // 4, pin_memory=True)
ms = timed(lambda: h.to(dev, non_blocking=False), iters)
out["results"].append({"name": "h2d_tensor_copy", "unit": "GB/s", "value": h.numel() * 4 / (ms * 1e6), "iterations": iters})
if inference:
    model = torch.nn.Sequential(torch.nn.Linear(1024, 4096), torch.nn.GELU(), torch.nn.Linear(4096, 1024)).to(dev).half().eval()
    for bs in (1, 64):
        inp = torch.randn(bs, 1024, device=dev, dtype=torch.float16)
        with torch.inference_mode():
            ms = timed(lambda: model(inp), iters)
        out["results"].append({"name": f"mlp_inference_bs{bs}_latency", "unit": "ms", "value": ms, "iterations": iters,
                               "detail": "synthetic 2-layer MLP (1024-4096-1024, fp16), random weights"})
        out["results"].append({"name": f"mlp_inference_bs{bs}_throughput", "unit": "samples/s", "value": bs / (ms / 1000), "iterations": iters})
print(json.dumps(out))
"""
