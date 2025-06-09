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
            if (scanf("%d", &MatrixCount) == 1 && MatrixCount >= 2)
            {
                f32 **Matrices = AllocateCPU(f32 *, MatrixCount);
                u32 *Heights = AllocateCPU(u32, MatrixCount);
                u32 *Widths = AllocateCPU(u32, MatrixCount);

                u32 I;
                for (I = 0; I < MatrixCount; I++)
                {
                    Matrices[I] = ReadMatrix(&Heights[I], &Widths[I]);
                    if (!Matrices[I])
                    {
                        break;
                    }
                }

                if (I == MatrixCount)
                {
                    u32 J;
                    for (J = 0; J < MatrixCount - 1; J++)
                    {
                        if (Widths[J] != Heights[J + 1])
                        {
                            fprintf(stderr, "Error: Matrix dimensions incompatible at position %d: (%dx%d) * (%dx%d)\n",
                                    J, Heights[J], Widths[J], Heights[J + 1], Widths[J + 1]);
                            ExitCode = 1;
                            break;
                        }
                    }

                    if (J == MatrixCount - 1)
                    {
                        u32 CurrentHeight = Heights[0];
                        u32 CurrentWidth = Widths[0];
                        f32 *CurrentResult = AllocateCPU(f32, CurrentHeight * CurrentWidth);
                        memcpy(CurrentResult, Matrices[0], sizeof(f32) * CurrentHeight * CurrentWidth);

                        for (u32 K = 1; K < MatrixCount; K++)
                        {
                            u32 NextHeight = Heights[K];
                            u32 NextWidth = Widths[K];
                            u32 ResultHeight = CurrentHeight;
                            u32 ResultWidth = NextWidth;

                            f32 *NewResult = AllocateCPU(f32, ResultHeight * ResultWidth);

                            u32 SizeA = sizeof(f32) * CurrentHeight * CurrentWidth;
                            u32 SizeB = sizeof(f32) * NextHeight * NextWidth;
                            u32 SizeC = sizeof(f32) * ResultHeight * ResultWidth;

                            f32 *DeviceA, *DeviceB, *DeviceC;
                            cudaMalloc(&DeviceA, SizeA);
                            cudaMalloc(&DeviceB, SizeB);
                            cudaMalloc(&DeviceC, SizeC);

                            cudaMemcpy(DeviceA, CurrentResult, SizeA, cudaMemcpyHostToDevice);
                            cudaMemcpy(DeviceB, Matrices[K], SizeB, cudaMemcpyHostToDevice);
                            cudaMemset(DeviceC, 0, SizeC);

                            if (strcmp(Algorithm, "simple") == 0)
                            {
                                dim3 ThreadsPerBlock(16, 16);
                                dim3 BlocksPerGrid((ResultWidth + 15) / 16, (ResultHeight + 15) / 16);
                                MatMulKernel<<<BlocksPerGrid, ThreadsPerBlock>>>(
                                    DeviceA, DeviceB, DeviceC, CurrentHeight, CurrentWidth, NextWidth);
                            }
                            else
                            {
                                dim3 ThreadsPerBlock(TILE_WIDTH, TILE_WIDTH);
                                dim3 BlocksPerGrid((ResultWidth + TILE_WIDTH - 1) / TILE_WIDTH,
                                                   (ResultHeight + TILE_WIDTH - 1) / TILE_WIDTH);
                                TiledMatmulKernel<<<BlocksPerGrid, ThreadsPerBlock>>>(
                                    DeviceA, DeviceB, DeviceC, CurrentHeight, CurrentWidth, NextWidth);
                            }

                            cudaMemcpy(NewResult, DeviceC, SizeC, cudaMemcpyDeviceToHost);

                            cudaFree(DeviceA);
                            cudaFree(DeviceB);
                            cudaFree(DeviceC);

                            free(CurrentResult);
                            CurrentResult = NewResult;
                            CurrentHeight = ResultHeight;
                            CurrentWidth = ResultWidth;
                        }

                        fprintf(stdout, "1\n");
                        PrintMatrix(CurrentResult, CurrentHeight, CurrentWidth);

                        free(CurrentResult);
                    }
                }

                for (u32 K = 0; K < I; K++)
                {
                    if (Matrices[K])
                    {
                        free(Matrices[K]);
                    }
                }
                free(Matrices);
                free(Heights);
                free(Widths);

                if (I < MatrixCount)
                {
                    fprintf(stderr, "Error: Failed to read matrix %d\n", I + 1);
                    ExitCode = 1;
                }
            }
            else
            {
                fprintf(stderr, "Error: Expected at least 2 matrices for multiplication\n");
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
        fprintf(stderr, "\nPerforms chain multiplication: A*B*C*... = ((A*B)*C)*...\n");
        ExitCode = 1;
    }

    return ExitCode;
}
