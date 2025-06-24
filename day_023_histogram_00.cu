#define LEET_GPU_NO_IMPORT
#ifndef LEET_GPU_NO_IMPORT
#include "solve.h"
#include <cuda_runtime.h>
#endif

// Runs with <<<1,1>>>
__global__ void Histogram(const int *input, int *histogram, int N, int num_bins)
{
    for (int i = 0; i < N; i++)
    {
        int bin = input[i];
        histogram[bin]++;
    }
}


// input, histogram are device pointers
void solve(const int* input, int* histogram, int N, int num_bins)
{
    Histogram<<<1,1>>>(input, histogram, N, num_bins);
}
