/*
 * Day 010: Vector Sum CLI (vecsum)
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
#include "day_010_reduction_kernel.cu"

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

    if (ArgumentCount == 1)
    {
        u32 MatrixCount;
        if (scanf("%d", &MatrixCount) == 1 && MatrixCount >= 1)
        {
            fprintf(stdout, "%d\n", MatrixCount);

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
                for (u32 J = 0; J < MatrixCount; J++)
                {
                    u32 Height = Heights[J];
                    u32 Width = Widths[J];
                    u32 VectorSize = Height * Width;
                    u32 Size = sizeof(f32) * VectorSize;

                    // Calculate grid dimensions
                    u32 BlockSize = BLOCK_SIZE;
                    u32 GridSize = (VectorSize + BlockSize - 1) / BlockSize;

                    f32 *DeviceInput, *DevicePartialSums;
                    cudaMalloc(&DeviceInput, Size);
                    cudaMalloc(&DevicePartialSums, sizeof(f32) * GridSize);

                    cudaMemcpy(DeviceInput, Matrices[J], Size, cudaMemcpyHostToDevice);

                    // Launch reduction kernel
                    ReduceVector<<<GridSize, BlockSize>>>(DeviceInput, DevicePartialSums, VectorSize);

                    // Copy partial sums back to host
                    f32 *PartialSums = AllocateCPU(f32, GridSize);
                    cudaMemcpy(PartialSums, DevicePartialSums, sizeof(f32) * GridSize, cudaMemcpyDeviceToHost);

                    // Sum partial results on CPU
                    f32 FinalSum = 0.0f;
                    for (u32 K = 0; K < GridSize; K++)
                    {
                        FinalSum += PartialSums[K];
                    }

                    // Output as 1x1 matrix
                    f32 *Result = AllocateCPU(f32, 1);
                    Result[0] = FinalSum;
                    PrintMatrix(Result, 1, 1);

                    cudaFree(DeviceInput);
                    cudaFree(DevicePartialSums);
                    free(PartialSums);
                    free(Result);
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
            fprintf(stderr, "Error: Expected at least 1 matrix for vector sum\n");
            ExitCode = 1;
        }
    }
    else
    {
        fprintf(stderr, "Usage: %s\n", Arguments[0]);
        fprintf(stderr, "\nReads matrix data from stdin in format:\n");
        fprintf(stderr, "  <matrix_count>\n");
        fprintf(stderr, "  <height1> <width1>\n");
        fprintf(stderr, "  <matrix1_data>\n");
        fprintf(stderr, "  <height2> <width2>\n");
        fprintf(stderr, "  <matrix2_data>\n");
        fprintf(stderr, "  ...\n");
        fprintf(stderr, "\nOutputs vector sums as 1x1 matrices\n");
        ExitCode = 1;
    }

    return ExitCode;
}
