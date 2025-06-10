/*
 * Day 009: Transpose a matrix
 *
 * Based on challege from LeetGPU
 */
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>

typedef float f32;
typedef uint32_t u32;
typedef uint64_t u64;

typedef int32_t s32;

__global__ void TransposeMatrix(const f32 *Input, f32 *Output, u32 Height, u32 Width)
{
    u32 Row = blockDim.y * blockIdx.y + threadIdx.y;
    u32 Col = blockDim.x * blockIdx.x + threadIdx.x;

    if (Row < Height && Col < Width)
    {
        Output[Row * Width + Col] = Input[Col * Height + Row];
    }
}
