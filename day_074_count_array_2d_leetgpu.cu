#include <cuda_runtime.h>

__global__ void count_2d_equal_kernel(const int *input, int *output, int N,
                                      int M, int K) {
  int Row = blockDim.x * blockIdx.x + threadIdx.x;
  int Col = blockDim.y * blockIdx.y + threadIdx.y;

  if (Row < N && Col < M) {
    int Pos = Row * M + Col;
    if (input[Pos] == K) {
      atomicAdd(output, 1);
    }
  }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int *input, int *output, int N, int M, int K) {
  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

  count_2d_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, M,
                                                            K);
  cudaDeviceSynchronize();
}
