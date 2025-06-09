/*
 * Day 008: Matrix Generation CLI (matgen)
 */
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef float f32;
typedef uint8_t u8;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int32_t s32;

#include "day_001_macros.h"
#include "day_008_generate_random_matrix.cu"
#include "day_008_generate_sequential_matrix.cu"

void PrintUsage(char *ProgramName)
{
    fprintf(stderr, "Usage: %s <algorithm> <seed> <count> <height1> <width1> [height2] [width2] ...\n", ProgramName);
    fprintf(stderr, "  algorithm: uniform (generates values in [0,1])\n");
    fprintf(stderr, "       seed: random seed for reproducible generation\n");
    fprintf(stderr, "      count: number of matrices to generate\n");
    fprintf(stderr, "    heightN: height of matrix N\n");
    fprintf(stderr, "     widthN: width of matrix N\n");
    fprintf(stderr, "\nExample: %s uniform 12345 2 10 20 20 10\n", ProgramName);
}

enum algo
{
    UNKOWN = 0x0,
    UNIFORM = 0x1,
    SEQUENTIAL = 0x2,
};

int main(int ArgumentCount, char *Arguments[])
{
    if (ArgumentCount < 6)
    {
        PrintUsage(Arguments[0]);
        return 1;
    }

    char *Algorithm = Arguments[1];
    u64 Seed;
    u32 MatrixCount;
    sscanf(Arguments[2], "%ld", &Seed);
    sscanf(Arguments[3], "%d", &MatrixCount);

    algo Algo = UNKOWN;

    if (strcmp(Algorithm, "uniform") == 0)
    {
        Algo = UNIFORM;
    }
    if (strcmp(Algorithm, "seq") == 0)
    {
        Algo = SEQUENTIAL;
    }
    if (Algo == UNKOWN)
    {
        fprintf(stderr, "Error: Only 'uniform' algorithm is supported\n");
        return 1;
    }

    if (ArgumentCount != (s32)(4 + 2 * MatrixCount))
    {
        fprintf(stderr, "Error: Expected %d arguments for %d matrices, got %d\n", 4 + 2 * MatrixCount, MatrixCount,
                ArgumentCount);
        return 1;
    }

    fprintf(stdout, "%d\n", MatrixCount);

    for (u32 MatrixIndex = 0; MatrixIndex < MatrixCount; MatrixIndex++)
    {
        u32 Height, Width;
        sscanf(Arguments[4 + MatrixIndex * 2], "%d", &Height);
        sscanf(Arguments[5 + MatrixIndex * 2], "%d", &Width);

        u32 SizeInBytes = sizeof(f32) * Width * Height;

        f32 *HostMatrix = AllocateCPU(f32, Width * Height);

        f32 *DeviceMatrix;
        cudaMalloc(&DeviceMatrix, SizeInBytes);

        dim3 ThreadsPerBlock(32);
        dim3 BlocksPerGrid((Width * Height + 32 - 1) / 32);

        if (Algo == UNIFORM)
        {
            GenerateRandomMatrix<<<BlocksPerGrid, ThreadsPerBlock>>>(DeviceMatrix, Width, Height, Seed + MatrixIndex);
        }
        if (Algo == SEQUENTIAL)
        {
            GenerateSequentialMatrix<<<BlocksPerGrid, ThreadsPerBlock>>>(DeviceMatrix, Width, Height);
        }

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

        free(HostMatrix);
        cudaFree(DeviceMatrix);
    }

    return 0;
}
