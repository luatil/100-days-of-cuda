#include "solve.h"
#include <cuda_runtime.h>

#define BLOCK_DIM 1024
#define COARSE_FACTOR 4

__global__ void ReduceKernel(const float *Input, float *Output, int N)
{
    __shared__ float Shared[BLOCK_DIM];

    const int TID = COARSE_FACTOR * blockDim.x * blockIdx.x + threadIdx.x;
    const int TX = threadIdx.x;

    Shared[TX] = 0.0f;
    for (int I = 0; I < COARSE_FACTOR; I++)
    {
        if (TID + blockDim.x * I < N)
        {
            Shared[TX] += Input[TID + blockDim.x * I];
        }
    }
    __syncthreads();

    for (int Stride = blockDim.x / 2; Stride > 0; Stride /= 2)
    {
        if (TX < Stride)
        {
            Shared[TX] += Shared[TX + Stride];
        }
        __syncthreads();
    }

    if (TX == 0)
    {
        atomicAdd(Output, Shared[0]);
    }
}

// input, output are device pointers
void Solve(const float *Input, float *Output, int N)
{
    const int BlockDim = BLOCK_DIM;
    const int GRID_DIM = (N + BlockDim - 1) / BlockDim;
    ReduceKernel<<<GRID_DIM, BlockDim>>>(Input, Output, N);
}
