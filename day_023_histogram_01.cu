#define LEET_GPU_NO_IMPORT
#ifndef LEET_GPU_NO_IMPORT
#include "solve.h"
#include <cuda_runtime.h>
#endif

#define BLOCK_SIZE 256

__global__ void Histogram(const int *input, int *histogram, int N, int num_bins)
{
    int Tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (Tid < N)
    {
        atomicAdd(&histogram[input[Tid]], 1);
    }
}


// input, histogram are device pointers
void solve(const int* input, int* histogram, int N, int num_bins)
{
    int GridDim = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    Histogram<<<GridDim, BLOCK_SIZE>>>(input, histogram, N, num_bins);
}
