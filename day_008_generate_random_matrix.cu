/*
 * Day 008: Generate Random Matrix
 */
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdint.h>
#include <stdlib.h>

typedef float f32;
typedef uint32_t u32;
typedef uint64_t u64;

typedef int32_t s32;

__global__ void GenerateRandomMatrix(f32 *OutputMatrix, u32 Width, u32 Height, u64 Seed)
{
    u32 Tid = blockIdx.x * blockDim.x + threadIdx.x;
    u32 TotalElements = Width * Height;

    if (Tid < TotalElements)
    {
        curandState State;
        curand_init(Seed, Tid, 0, &State);
        OutputMatrix[Tid] = curand_uniform(&State); // Uniform [0,1]
    }
}
