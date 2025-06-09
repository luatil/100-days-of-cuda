/*
 * Day 008: Matrix Multiplication CLI (matmul)
 */
#include <cuda_runtime.h>
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
#include "day_002_simple_matmul_kernel.cu"
#include "day_007_tiled_matmul_kernel.cu"

f32 *ReadMatrix(u32 *Height, u32 *Width)
{
    if (scanf("%d %d", Height, Width) != 2)
    {
        return NULL;
    }

    f32 *Matrix = AllocateCPU(f32, (*Height) * (*Width));

    for (u32 I = 0; I < *Height; I++)
    {
        for (u32 J = 0; J < *Width; J++)
        {
            if (scanf("%f", &Matrix[I * (*Width) + J]) != 1)
            {
                free(Matrix);
                return NULL;
            }
        }
    }

    return Matrix;
}

void PrintMatrix(f32 *Matrix, u32 Height, u32 Width)
{
    fprintf(stdout, "%d %d\n", Height, Width);
    for (u32 I = 0; I < Height; I++)
    {
        for (u32 J = 0; J < Width; J++)
        {
            fprintf(stdout, "%.6f ", Matrix[I * Width + J]);
        }
        fprintf(stdout, "\n");
    }
}

int main(int ArgumentCount, char *Arguments[])
{
    int ExitCode = 0;

    if (ArgumentCount == 2)
    {
        char *Algorithm = Arguments[1];

        if (strcmp(Algorithm, "simple") == 0 || strcmp(Algorithm, "tiled") == 0)
        {
            u32 MatrixCount;
            if (scanf("%d", &MatrixCount) == 1 && MatrixCount == 2)
            {
                u32 HeightA, WidthA, HeightB, WidthB;
                f32 *HostA = ReadMatrix(&HeightA, &WidthA);
                f32 *HostB = ReadMatrix(&HeightB, &WidthB);

                if (HostA && HostB)
                {
                    if (WidthA == HeightB)
                    {
                        u32 HeightC = HeightA;
                        u32 WidthC = WidthB;
                        f32 *HostC = AllocateCPU(f32, HeightC * WidthC);

                        u32 SizeA = sizeof(f32) * HeightA * WidthA;
                        u32 SizeB = sizeof(f32) * HeightB * WidthB;
                        u32 SizeC = sizeof(f32) * HeightC * WidthC;

                        f32 *DeviceA, *DeviceB, *DeviceC;
                        cudaMalloc(&DeviceA, SizeA);
                        cudaMalloc(&DeviceB, SizeB);
                        cudaMalloc(&DeviceC, SizeC);

                        cudaMemcpy(DeviceA, HostA, SizeA, cudaMemcpyHostToDevice);
                        cudaMemcpy(DeviceB, HostB, SizeB, cudaMemcpyHostToDevice);
                        cudaMemset(DeviceC, 0, SizeC);

                        if (strcmp(Algorithm, "simple") == 0)
                        {
                            dim3 ThreadsPerBlock(16, 16);
                            dim3 BlocksPerGrid((WidthC + 15) / 16, (HeightC + 15) / 16);
                            MatMulKernel<<<BlocksPerGrid, ThreadsPerBlock>>>(DeviceA, DeviceB, DeviceC, HeightA, WidthA,
                                                                             WidthB);
                        }
                        else
                        {
                            dim3 ThreadsPerBlock(TILE_WIDTH, TILE_WIDTH);
                            dim3 BlocksPerGrid((WidthC + TILE_WIDTH - 1) / TILE_WIDTH,
                                               (HeightC + TILE_WIDTH - 1) / TILE_WIDTH);
                            TiledMatmulKernel<<<BlocksPerGrid, ThreadsPerBlock>>>(DeviceA, DeviceB, DeviceC, HeightA,
                                                                                  WidthA, WidthB);
                        }

                        cudaMemcpy(HostC, DeviceC, SizeC, cudaMemcpyDeviceToHost);

                        fprintf(stdout, "1\n");
                        PrintMatrix(HostC, HeightC, WidthC);

                        free(HostC);
                        cudaFree(DeviceA);
                        cudaFree(DeviceB);
                        cudaFree(DeviceC);
                    }
                    else
                    {
                        fprintf(stderr, "Error: Matrix dimensions incompatible for multiplication (%dx%d) * (%dx%d)\n",
                                HeightA, WidthA, HeightB, WidthB);
                        ExitCode = 1;
                    }

                    if (HostA)
                    {
                        free(HostA);
                    }
                    if (HostB)
                    {
                        free(HostB);
                    }
                }
                else
                {
                    fprintf(stderr, "Error: Failed to read matrices\n");
                    if (HostA)
                    {
                        free(HostA);
                    }
                    if (HostB)
                    {
                        free(HostB);
                    }
                    ExitCode = 1;
                }
            }
            else
            {
                fprintf(stderr, "Error: Expected exactly 2 matrices for multiplication\n");
                ExitCode = 1;
            }
        }
        else
        {
            fprintf(stderr, "Error: Algorithm must be 'simple' or 'tiled'\n");
            ExitCode = 1;
        }
    }
    else
    {
        fprintf(stderr, "Usage: %s <algorithm>\n", Arguments[0]);
        fprintf(stderr, "  algorithm: simple or tiled\n");
        fprintf(stderr, "\nReads matrix data from stdin in format:\n");
        fprintf(stderr, "  <matrix_count>\n");
        fprintf(stderr, "  <height1> <width1>\n");
        fprintf(stderr, "  <matrix1_data>\n");
        fprintf(stderr, "  <height2> <width2>\n");
        fprintf(stderr, "  <matrix2_data>\n");
        fprintf(stderr, "  ...\n");
        ExitCode = 1;
    }

    return ExitCode;
}
