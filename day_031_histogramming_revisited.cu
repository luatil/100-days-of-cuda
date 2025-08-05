#define LEET_GPU_NOIMPORT
#define NO_GPU

#ifndef LEET_GPU_NOIMPORT
#include "solve.h"
#include <cuda_runtime.h>
#endif
#include <cfloat>
#include <cmath>

#ifdef NO_GPU
#define GLOBAL
#define SHARED
#define SYNCTHREADS()
#define ATOMIC_ADD(_a, _b)

struct dim3
{
    int x;
    int y;
    int z;
};

dim3 ThreadIdx;
dim3 BlockDim;
dim3 BlockIdx;
#endif

typedef unsigned int u32;

#define MAX_NUM_BINS 1024

GLOBAL void HistogramKernel(const int *Input, int *Histogram, int N, int NumBins)
{
    SHARED int Shared[MAX_NUM_BINS];

    for (int I = threadIdx.x; I < NumBins; I += blockDim.x)
    {
        Shared[I] = 0;
    }
    SYNCTHREADS();

    u32 Tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (Tid < N)
    {
        int Pos = Input[Tid];
        ATOMIC_ADD(Shared + Pos, 1);
    }
    SYNCTHREADS();

    for (int Bin = threadIdx.x; Bin < NumBins; Bin += blockDim.x)
    {
        int BinValue = Shared[Bin];
        if (BinValue > 0)
        {
            ATOMIC_ADD(histogram + bin, BinValue);
        }
    }
}

#ifndef LEET_GPU_NOIMPORT
// input, histogram are device pointers
void solve(const int *input, int *histogram, int N, int num_bins)
{
    const int BlockDim = 256;
    const int GridDim = (N + BlockDim - 1) / BlockDim;
    HistogramKernel<<<GridDim, BlockDim>>>(input, histogram, N, num_bins);
}
#endif