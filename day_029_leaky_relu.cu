#include "solve.h"
#include <cuda_runtime.h>

__global__ void LeakyReluKernel(const float *Input, float *Output, int N)
{
    float Alpha = 0.01;
    int Tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (Tid < N)
    {
        float X = Input[Tid];
        if (X > 0.0f)
        {
            Output[Tid] = X;
        }
        else if (X <= 0.0f)
        {
            Output[Tid] = Alpha * X;
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void Solve(const float *Input, float *Output, int N)
{
    int ThreadsPerBlock = 256;
    int BlocksPerGrid = (N + ThreadsPerBlock - 1) / ThreadsPerBlock;

    LeakyReluKernel<<<BlocksPerGrid, ThreadsPerBlock>>>(Input, Output, N);
    cudaDeviceSynchronize();
}
