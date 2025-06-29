#include "solve.h"
#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* input, float* output, int N) {
    float alpha = 0.01;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < N) {
        float x = input[tid];
        if (x > 0.0f) {
            output[tid] = x;
        } else if (x <= 0.0f) {
            output[tid] = alpha * x;
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    leaky_relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
