#include "solve.h"
#include <cuda_runtime.h>

#define BLOCK_DIM 128

__global__ void MonteCarloIntegrationKernel(const float *y_samples, float *result, float a, float b, int n_samples)
{
    int Tid = blockDim.x * blockIdx.x + threadIdx.x;
    int Tx = threadIdx.x;

    __shared__ float Shared[BLOCK_DIM];
    Shared[Tx] = (Tid < n_samples) ? y_samples[Tid] : 0.0f;
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
        atomicAdd(result, (b - a) * (Shared[0] / n_samples));
    }
}

// y_samples, result are device pointers
void solve(const float *y_samples, float *result, float a, float b, int n_samples)
{
    int BlockDim = BLOCK_DIM;
    int GridDim = (n_samples + BlockDim - 1) / BlockDim;
    MonteCarloIntegrationKernel<<<GridDim, BlockDim>>>(y_samples, result, a, b, n_samples);
}
