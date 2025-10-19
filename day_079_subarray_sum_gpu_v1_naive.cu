#include "day_079_subarray_sum.cuh"
#include <cuda_runtime.h>

__global__ void subarray_sum_kernel(const int* input, int* output, int N, int S, int E) {
  // Simple parallel reduction using shared memory
  extern __shared__ int sdata[];

  int tid = threadIdx.x;
  int idx = S + blockIdx.x * blockDim.x + threadIdx.x;

  // Load data into shared memory
  sdata[tid] = (idx <= E) ? input[idx] : 0;
  __syncthreads();

  // Reduction in shared memory
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s && idx + s <= E) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // Write result for this block to global memory
  if (tid == 0) {
    atomicAdd(output, sdata[0]);
  }
}

extern "C" void solve_subarray_sum_gpu_v1_naive(const int* input, int* output, int N, int S, int E) {
  int* d_input;
  int* d_output;
  int size = E - S + 1;

  // Allocate device memory
  cudaMalloc(&d_input, N * sizeof(int));
  cudaMalloc(&d_output, sizeof(int));

  // Copy input to device
  cudaMemcpy(d_input, input, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_output, output, sizeof(int), cudaMemcpyHostToDevice);

  // Launch kernel
  int threads = 256;
  int blocks = (size + threads - 1) / threads;
  int shared_mem_size = threads * sizeof(int);

  subarray_sum_kernel<<<blocks, threads, shared_mem_size>>>(d_input, d_output, N, S, E);

  // Copy result back to host
  cudaMemcpy(output, d_output, sizeof(int), cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_input);
  cudaFree(d_output);
}
