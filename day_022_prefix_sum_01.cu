#ifndef LEET_GPU_NO_IMPORT
#include "solve.h"
#include <cuda_runtime.h>
#endif

__global__ void PrefixSum(const float *Input, float *Output, int N)
{
    Output[0] = Input[0];
    for (int I = 1; I < N; I++)
    {
        Output[I] = Output[I - 1] + Input[I];
    }
}

// input, output are device pointers
void solve(const float *input, float *output, int N)
{
    PrefixSum<<<1, 1>>>(input, output, N);
}
