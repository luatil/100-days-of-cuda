#include "solve.h"
#include <cuda_runtime.h>

#define BLOCK_DIM 128

__global__ void MonteCarloIntegrationKernel(const float *YSamples, float *Result, float A, float B, int NSamples)
{
    int Tid = blockDim.x * blockIdx.x + threadIdx.x;
    int Tx = threadIdx.x;

    __shared__ float Shared[BLOCK_DIM];
    Shared[Tx] = (Tid < NSamples) ? YSamples[Tid] : 0.0f;
    __syncthreads();

    for (int Stride = blockDim.x / 2; Stride > 0; Stride >>= 1)
    {
        float Temp = 0.0f;
        if (Tx < Stride)
        {
            Temp = Shared[Tx] + Shared[Tx + Stride];
        }
        __syncthreads();
        if (Tx < Stride)
        {
            Shared[Tx] = Temp;
        }
        __syncthreads();
    }

    if (Tx == 0)
    {
        atomicAdd(Result, (B - A) * (Shared[0] / NSamples));
    }
}

// y_samples, result are device pointers
void Solve(const float *YSamples, float *Result, float A, float B, int NSamples)
{
    int BlockDim = BLOCK_DIM;
    int GridDim = (NSamples + BlockDim - 1) / BlockDim;
    MonteCarloIntegrationKernel<<<GridDim, BlockDim>>>(YSamples, Result, A, B, NSamples);
}
