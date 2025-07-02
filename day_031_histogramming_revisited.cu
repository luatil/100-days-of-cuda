#define LEET_GPU_NOIMPORT
#define NO_GPU

#ifndef LEET_GPU_NOIMPORT
#include "solve.h"
#include <cuda_runtime.h>
#endif
#include <cmath>
#include <cfloat>

#ifdef NO_GPU
#define __global__ 
#define __shared__
#define __syncthreads()
#define atomicAdd(_a, _b)

struct dim3 {
    int x;
    int y;
    int z;
};

dim3 threadIdx;
dim3 blockDim;
dim3 blockIdx;
#endif

typedef unsigned int u32;

#define MAX_NUM_BINS 1024

__global__ void HistogramKernel(const int* input, int* histogram, int N, int num_bins) 
{
    __shared__ int Shared[MAX_NUM_BINS];

    for (int i = threadIdx.x; i < num_bins; i += blockDim.x)
    {
        Shared[i] = 0;
    }
    __syncthreads();

    u32 Tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (Tid < N)
    {
        int Pos = input[Tid];
        atomicAdd(Shared + Pos, 1); 
    }
    __syncthreads();

    for (int bin = threadIdx.x; bin < num_bins; bin += blockDim.x)
    {
        int BinValue = Shared[bin];
        if (BinValue > 0) {
            atomicAdd(histogram + bin, BinValue);
        }
    }
}

#ifndef LEET_GPU_NOIMPORT
// input, histogram are device pointers
void solve(const int* input, int* histogram, int N, int num_bins) {
    const int BlockDim = 256;
    const int GridDim = (N + BlockDim - 1) / BlockDim;
    HistogramKernel<<<GridDim, BlockDim>>>(input, histogram, N, num_bins);
}
#endif