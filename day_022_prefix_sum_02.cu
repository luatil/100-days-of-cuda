#define LEET_GPU_NO_IMPORT
#ifndef LEET_GPU_NO_IMPORT
#include "solve.h"
#include <cuda_runtime.h>
#endif

#define BLOCK_DIM 256
#define DEBUG 0
#define DEBUG_PARTIAL_SUMS 0

__global__ void CalculatePartialSums(const float *Input, float *Output, float *PartialSums, int N)
{
    __shared__ float Shared[BLOCK_DIM];

    // NOTE(luatil): Could do coarsening here
    int Tid = blockDim.x * blockIdx.x + threadIdx.x;
    int Tx = threadIdx.x;

    Shared[Tx] = Tid < N ? Input[Tid] : 0.0f;
    __syncthreads();

    for (int Stride = 1; Stride <= blockDim.x / 2; Stride *= 2)
    {
        float Temp = 0.0f;
        if (Tx >= Stride)
        {
            Temp = Shared[Tx] + Shared[Tx - Stride];
        }
        __syncthreads();
        if (Tx >= Stride)
        {
            Shared[Tx] = Temp;
        }
        __syncthreads();
    }

#if DEBUG
    printf("Shared[%d]=%.3f\n", Tx, Shared[Tx]);
    printf("Input [%d]=%.3f\n", Tid, Input[Tid]);
#endif

    if (Tid < N)
    {
        Output[Tid] = Shared[Tx];
    }

    if (Tx == BLOCK_DIM - 1)
    {
        PartialSums[blockIdx.x] = Shared[Tx];
    }
}

__global__ void ScanPartialSums(float *PartialSums, int N)
{
#if DEBUG_PARTIAL_SUMS
    for (int I = 0; I < N; I++)
    {
        printf("PartialSums[%d] = %.3f\n", I, PartialSums[I]);
    }
#endif
    for (int I = 1; I < N; I++)
    {
        PartialSums[I] += PartialSums[I - 1];
    }
#if DEBUG_PARTIAL_SUMS
    for (int I = 0; I < N; I++)
    {
        printf("PartialSums[%d] = %.3f\n", I, PartialSums[I]);
    }
#endif
}

__global__ void ExpandPartialSums(const float *PartialSums, float *Output, int N)
{
    const int IS_FIRST_BLOCK = blockIdx.x == 0;
    const int TID = blockDim.x * blockIdx.x + threadIdx.x;
    const int TID_IN_RANGE = TID < N;

    __shared__ float BlockPartialSum;

    if (!IS_FIRST_BLOCK)
    {
        if (threadIdx.x == 0)
        {
            BlockPartialSum = PartialSums[blockIdx.x - 1];
        }
        __syncthreads();

        if (TID_IN_RANGE)
        {
            Output[TID] += BlockPartialSum;
        }
    }
}

// input, output are device pointers
void Solve(const float *Input, float *Output, int N)
{
    const int NUMBER_OF_BLOCKS = (N + BLOCK_DIM - 1) / BLOCK_DIM;

    float *PartialSums;
    cudaMalloc(&PartialSums, NUMBER_OF_BLOCKS * sizeof(float));

    CalculatePartialSums<<<NUMBER_OF_BLOCKS, BLOCK_DIM>>>(Input, Output, PartialSums, N);
    ScanPartialSums<<<1, 1>>>(PartialSums, NUMBER_OF_BLOCKS);
    // Could optimize by removing first block.
    ExpandPartialSums<<<NUMBER_OF_BLOCKS, BLOCK_DIM>>>(PartialSums, Output, N);

    cudaFree(PartialSums);
}
