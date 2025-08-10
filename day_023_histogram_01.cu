#define LEET_GPU_NO_IMPORT
#ifndef LEET_GPU_NO_IMPORT
#include "solve.h"
#include <cuda_runtime.h>
#endif

#define BLOCK_SIZE 256

__global__ void HistogramKernel(const int *Input, int *Histogram, int N, int NumBins)
{
    int Tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (Tid < N)
    {
        atomicAdd(&Histogram[Input[Tid]], 1);
    }
}

// input, histogram are device pointers
void Solve(const int *Input, int *Histogram, int N, int NumBins)
{
    int GridDim = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    HistogramKernel<<<GridDim, BLOCK_SIZE>>>(Input, Histogram, N, NumBins);
}
