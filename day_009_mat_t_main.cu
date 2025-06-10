/*
 * Day 009: Matrix Transpose CLI (mat_t)
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
#include "day_009_matrix_transpose_kernel.cu"

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
                    u32 Size = sizeof(f32) * Height * Width;

                    f32 *DeviceInput, *DeviceOutput;
                    cudaMalloc(&DeviceInput, Size);
                    cudaMalloc(&DeviceOutput, Size);

                    cudaMemcpy(DeviceInput, Matrices[J], Size, cudaMemcpyHostToDevice);

                    dim3 ThreadsPerBlock(16, 16);
                    dim3 BlocksPerGrid((Width + 15) / 16, (Height + 15) / 16);
                    TransposeMatrix<<<BlocksPerGrid, ThreadsPerBlock>>>(DeviceInput, DeviceOutput, Height, Width);

                    f32 *TransposedMatrix = AllocateCPU(f32, Height * Width);
                    cudaMemcpy(TransposedMatrix, DeviceOutput, Size, cudaMemcpyDeviceToHost);

                    PrintMatrix(TransposedMatrix, Width, Height);

                    cudaFree(DeviceInput);
                    cudaFree(DeviceOutput);
                    free(TransposedMatrix);
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
            fprintf(stderr, "Error: Expected at least 1 matrix for transpose\n");
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
        fprintf(stderr, "\nOutputs transposed matrices in same format as matgen\n");
        ExitCode = 1;
    }

    return ExitCode;
}
