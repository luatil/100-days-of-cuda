#define LEET_GPU_NO_IMPORT
#ifndef LEET_GPU_NO_IMPORT
#include "solve.h"
#include <cuda_runtime.h>
#endif

// Runs with <<<1,1>>>
__global__ void Histogram(const int *Input, int *Histogram, int N, int NumBins)
{
    for (int I = 0; I < N; I++)
    {
        int Bin = Input[I];
        Histogram[Bin]++;
    }
}

// input, histogram are device pointers
void Solve(const int *Input, int *Histogram, int N, int NumBins)
{
    Histogram<<<1, 1>>>(Input, Histogram, N, NumBins);
}
