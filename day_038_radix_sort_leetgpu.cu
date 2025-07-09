// NOTE(luatil): Second solution to RadixSort still does not work
#define LEET_GPU_NO_IMPORT
#ifndef LEET_GPU_NO_IMPORT
#include "solve.h"
#include <cuda_runtime.h>
#endif

#define SOLUTION 1

#define BLOCK_SIZE 256
#define RADIX_BITS 4
#define RADIX_SIZE (1 << RADIX_BITS) // 16 for 4-bit radix

__device__ void Swap(unsigned int *A, int I, int J)
{
    unsigned int Temp = A[I];
    A[I] = A[J];
    A[J] = Temp;
}

#if SOLUTION == 0
__global__ void RadixSort(const unsigned int *input, unsigned int *output, int N)
{
    for (int i = 0; i < N; i++)
    {
        output[i] = input[i];
    }

    // NOTE(luatil): I can't believe it sorts algorithm:
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (output[i] < output[j])
            {
                Swap(output, i, j);
            }
        }
    }
}
#elif SOLUTION == 1
__global__ void RadixSort(const unsigned int *Input, unsigned int *Output, int N)
{
    int Tid = blockDim.x * blockIdx.x + threadIdx.x;
    int Tx = threadIdx.x;

    if (Tid < N)
    {
        Output[Tid] = Input[Tid];
    }
    __syncthreads();

    for (int Bit = 0; Bit < 32; Bit++)
    {
        __shared__ unsigned int Shared[BLOCK_SIZE];
        __shared__ int Zeros;
        __shared__ int Ones;

        if (Tx == 0)
        {
            Zeros = 0;
            Ones = 0;
        }
        __syncthreads();

        if (Tid < N)
        {
            Shared[Tx] = Output[Tid];
        }
        __syncthreads();

        if (Tid < N)
        {
            int BitVal = (Shared[Tx] >> Bit) & 1;
            if (BitVal == 0)
            {
                atomicAdd(&Zeros, 1);
            }
            else
            {
                atomicAdd(&Ones, 1);
            }
        }
        __syncthreads();

        // Partition based on bit value
        if (Tid < N)
        {
            int bit_val = (Shared[Tx] >> Bit) & 1;
            int pos = 0;

            if (bit_val == 0)
            {
                // Count zeros before this position
                for (int i = 0; i < Tx; i++)
                {
                    if (Tid - Tx + i < N)
                    {
                        int prev_bit = (Shared[i] >> Bit) & 1;
                        if (prev_bit == 0)
                            pos++;
                    }
                }
            }
            else
            {
                // Count ones before this position
                pos = Ones;
                for (int i = 0; i < Tx; i++)
                {
                    if (Tid - Tx + i < N)
                    {
                        int prev_bit = (Shared[i] >> Bit) & 1;
                        if (prev_bit == 1)
                            pos++;
                    }
                }
            }

            if (pos < N)
            {
                Output[blockIdx.x * blockDim.x + pos] = Shared[Tx];
            }
        }
        __syncthreads();
    }
}
#endif

// input, output are device pointers
void solve(const unsigned int *input, unsigned int *output, int N)
{
    int GridDim = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    RadixSort<<<GridDim, BLOCK_SIZE>>>(input, output, N);
}
