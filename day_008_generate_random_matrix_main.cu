/*
 * Day 008: Generate Random Matrix CLI
 */
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef float f32;
typedef uint8_t u8;
typedef uint32_t u32;
typedef uint64_t u64;

typedef int32_t s32;

#include "day_001_macros.h"
#include "day_008_generate_random_matrix.cu"

int main(int ArgumentCount, char *Arguments[])
{
    if (ArgumentCount == 4)
    {
        u32 Width, Height;
        u64 Seed;

        sscanf(Arguments[1], "%ld", &Seed);
        sscanf(Arguments[2], "%d", &Height);
        sscanf(Arguments[3], "%d", &Width);

        u32 SizeInBytes = sizeof(f32) * Width * Height;

        f32 *HostMatrix = AllocateCPU(f32, Width * Height);

        f32 *DeviceMatrix;
        cudaMalloc(&DeviceMatrix, SizeInBytes);

        dim3 ThreadsPerBlock(32);
        dim3 BlocksPerGrid((Width * Height + 32 - 1) / 32);

        GenerateRandomMatrix<<<BlocksPerGrid, ThreadsPerBlock>>>(DeviceMatrix, Width, Height, Seed);

        cudaMemcpy(HostMatrix, DeviceMatrix, SizeInBytes, cudaMemcpyDeviceToHost);

        fprintf(stdout, "%d %d\n", Height, Width);
        for (u32 I = 0; I < Height; I++)
        {
            for (u32 J = 0; J < Width; J++)
            {
                fprintf(stdout, "%.6f ", HostMatrix[I * Width + J]);
            }
            fprintf(stdout, "\n");
        }
    }
    else
    {
        fprintf(stderr, "Usage: %s [seed] [height] [width]\n", Arguments[0]);
        fprintf(stderr, "   seed: Seed to generate the Matrix\n");
        fprintf(stderr, " height: Matrix height (positive integer)\n");
        fprintf(stderr, "  width: Matrix width (positive integer)\n");
    }
}
