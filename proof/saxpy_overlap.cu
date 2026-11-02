// ryzen-sm86-hybrid proof: CPU-GPU pipelined overlap on unified memory.
//
// Hardware: AMD Ryzen 7 8700F (Zen4) + NVIDIA RTX 3090 (compute 8.6, SM86).
// Hypothesis under test (from the UM-coherence papers): managed pages are
// exclusively GPU-held during a kernel, so the CPU cannot touch them mid-kernel;
// but a *pipelined* schedule (CPU prep of batch i+1 issued while the GPU runs
// kernel i on a separate stream) overlaps CPU write and GPU compute.
//
// Compare:
//   serialized : cpu-prep(i) -> copy(i) -> gpu-kernel(i) -> sync(i).
//   pipelined  : cpu writes host (pinned) buffer for i+1 while stream[cur] runs
//                kernel(i), then an async H2D copy feeds a device buffer the
//                GPU owns exclusively. This is the design the UM-coherence
//                papers endorse: the CPU never touches memory the GPU is
//                reading concurrently; pinned host + async copy overlap for real.
#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>

#define CHECK(x)                                                        \
  do {                                                                  \
    cudaError_t e = (x);                                                \
    if (e != cudaSuccess) {                                             \
      fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e),      \
              __FILE__, __LINE__);                                      \
      return 1;                                                         \
    }                                                                   \
  } while (0)

__global__ void saxpy(const float* a, const float* x, float* y, int n, float alpha, int iters) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = alpha * a[i] + x[i];
    // Compute-bound inner loop so the GPU kernel dominates the CPU prep write,
    // the realistic case where pipelining (overlap) should be measurable.
    for (int it = 1; it < iters; it++) v = v * 1.0000001f + 0.0000001f;
    y[i] = v;
  }
}

// Fill the source arrays on the host. With managed memory, prefetching is a
// hint; the write itself is the CPU-side "prep" work our pipeline overlaps.
static void fill(float* a, float* x, int n) {
  for (int i = 0; i < n; i++) {
    a[i] = 1.0f;
    x[i] = (float)i;
  }
}

int main() {
  CHECK(cudaSetDevice(0));
  cudaDeviceProp prop{};
  CHECK(cudaGetDeviceProperties(&prop, 0));

  constexpr int N = 1 << 22;   // 4M floats = 16MB per vector
  constexpr int BATCHES = 24;
  constexpr int BLOCK = 256;
  constexpr int ITERS = 2000;   // inner-loop work per element (kernel-bound)
  constexpr float ALPHA = 2.0f;
  const size_t bytes = (size_t)N * sizeof(float);

  // Device buffers + pinned host buffers. The pinned host page is CPU-owned
  // (prep happens there); an async H2D copy hands it to the GPU exclusively.
  float *a0, *x0, *a1, *x1, *y0, *y1;
  float *ha0, *hx0, *ha1, *hx1;
  CHECK(cudaMalloc(&a0, bytes)); CHECK(cudaMalloc(&x0, bytes)); CHECK(cudaMalloc(&y0, bytes));
  CHECK(cudaMalloc(&a1, bytes)); CHECK(cudaMalloc(&x1, bytes)); CHECK(cudaMalloc(&y1, bytes));
  CHECK(cudaMallocHost(&ha0, bytes)); CHECK(cudaMallocHost(&hx0, bytes));
  CHECK(cudaMallocHost(&ha1, bytes)); CHECK(cudaMallocHost(&hx1, bytes));

  auto now = [] { return std::chrono::steady_clock::now(); };
  auto ms = [](auto s, auto e) {
    return std::chrono::duration<double, std::milli>(e - s).count();
  };

  // ---- Serialized: fill(pinned) -> memcpy D -> kernel -> sync ----
  auto t0 = now();
  for (int b = 0; b < BATCHES; b++) {
    float* A = (b % 2 == 0) ? a0 : a1;
    float* X = (b % 2 == 0) ? x0 : x1;
    float* Y = (b % 2 == 0) ? y0 : y1;
    float* HA = (b % 2 == 0) ? ha0 : ha1;
    float* HX = (b % 2 == 0) ? hx0 : hx1;
    fill(HA, HX, N);
    CHECK(cudaMemcpyAsync(A, HA, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyAsync(X, HX, bytes, cudaMemcpyHostToDevice));
    saxpy<<<(N + BLOCK - 1) / BLOCK, BLOCK>>>(A, X, Y, N, ALPHA, ITERS);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
  }
  double serial_ms = ms(t0, now());

  // ---- Pipelined: two streams, prep(i+1) issued once kernel(i) is launched ----
  cudaStream_t s2[2];
  CHECK(cudaStreamCreate(&s2[0]));
  CHECK(cudaStreamCreate(&s2[1]));
  // One event per lane: batch-b reuses lane b%2, so the next H2D copy into that
  // lane must wait for the previous kernel that read it (batch b-2). Without
  // this, the copy for batch i+1 could race the batch i-1 kernel still draining.
  cudaEvent_t ev[2];
  CHECK(cudaEventCreate(&ev[0]));
  CHECK(cudaEventCreate(&ev[1]));
  auto t1 = now();
  fill(ha0, hx0, N);                                 // prep batch 0 on host
  CHECK(cudaMemcpyAsync(a0, ha0, bytes, cudaMemcpyHostToDevice, s2[0]));
  CHECK(cudaMemcpyAsync(x0, hx0, bytes, cudaMemcpyHostToDevice, s2[0]));
  saxpy<<<(N + BLOCK - 1) / BLOCK, BLOCK, 0, s2[0]>>>(a0, x0, y0, N, ALPHA, ITERS);
  CHECK(cudaGetLastError());
  CHECK(cudaEventRecord(ev[0], s2[0]));
  for (int b = 1; b < BATCHES; b++) {
    int cur = b % 2;
    int nxt = 1 - cur;
    float* A = cur == 0 ? a0 : a1;
    float* X = cur == 0 ? x0 : x1;
    float* Y = cur == 0 ? y0 : y1;
    float* NA = nxt == 0 ? a0 : a1;
    float* NX = nxt == 0 ? x0 : x1;
    float* NY = nxt == 0 ? y0 : y1;
    float* HA = nxt == 0 ? ha0 : ha1;
    float* HX = nxt == 0 ? hx0 : hx1;
    fill(HA, HX, N);                                 // CPU prep i+1 in pinned host
    CHECK(cudaStreamWaitEvent(s2[nxt], ev[nxt]));    // lane v-1 kernel must drain
    CHECK(cudaMemcpyAsync(NA, HA, bytes, cudaMemcpyHostToDevice, s2[nxt]));
    CHECK(cudaMemcpyAsync(NX, HX, bytes, cudaMemcpyHostToDevice, s2[nxt]));
    saxpy<<<(N + BLOCK - 1) / BLOCK, BLOCK, 0, s2[cur]>>>(A, X, Y, N, ALPHA, ITERS);
    (void)NY;
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(ev[cur], s2[cur]));
  }
  CHECK(cudaStreamSynchronize(s2[0]));
  CHECK(cudaStreamSynchronize(s2[1]));
  double pipe_ms = ms(t1, now());

  cudaStreamDestroy(s2[0]);
  cudaStreamDestroy(s2[1]);
  cudaEventDestroy(ev[0]);
  cudaEventDestroy(ev[1]);
  cudaFree(a0); cudaFree(x0); cudaFree(y0);
  cudaFree(a1); cudaFree(x1); cudaFree(y1);
  cudaFreeHost(ha0); cudaFreeHost(hx0); cudaFreeHost(ha1); cudaFreeHost(hx1);

  printf("{\"device\":\"%s\",\"cc\":\"%d.%d\",\"sm\":%d,\"batches\":%d,\"n\":%d,"
         "\"iters\":%d,\"serial_ms\":%.2f,\"pipelined_ms\":%.2f,\"speedup\":%.3f}\n",
         prop.name, prop.major, prop.minor, prop.multiProcessorCount, BATCHES, N,
         ITERS, serial_ms, pipe_ms, serial_ms / pipe_ms);
  return 0;
}