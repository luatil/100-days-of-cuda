/*
 * Day 06: Thread Synchronization - Global Array Reverse
 *
 * Based on chapter 4 from PMPP.
 *
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

typedef float f32;
typedef unsigned char u8;
typedef unsigned int u32;
typedef int s32;

__global__ void ReverseArray(u32 *Input, u32 *Output, u32 N)
{
    extern __shared__ u32 SharedData[];

    u32 ThreadIdx = threadIdx.x;
    u32 BlockSize = blockDim.x;
    u32 BlockId = blockIdx.x;

    // Calculate which "chunk" this block should process
    // But we process chunks from opposite ends of the array
    u32 SourceBlockStart = BlockId * BlockSize;
    u32 DestBlockStart = N - (BlockId + 1) * BlockSize;

    u32 SourceIdx = SourceBlockStart + ThreadIdx;
    u32 DestIdx = DestBlockStart + (BlockSize - 1 - ThreadIdx);

    // Load data into shared memory
    if (SourceIdx < N)
    {
        SharedData[ThreadIdx] = Input[SourceIdx];
    }

    // Synchronize to ensure all threads have loaded their data
    __syncthreads();

    // Write to output in reversed order
    if (DestIdx < N && SourceIdx < N)
    {
        Output[DestIdx] = SharedData[ThreadIdx];
    }
}
