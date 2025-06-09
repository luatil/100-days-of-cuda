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

#include "day_001_macros.h"
#include "day_006_reverse_array_kernel.cu"

int main()
{
    const u32 N = 1024;
    const u32 THREADS_PER_BLOCK = 256;
    const u32 BLOCKS_PER_GRID = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const u32 SHARED_DATA_SIZE = THREADS_PER_BLOCK * sizeof(u32);

    u32 *HostInput = AllocateCPU(u32, N);
    u32 *HostOutput = AllocateCPU(u32, N);

    for (u32 I = 0; I < N; I++)
    {
        HostInput[I] = I;
    }

    u32 *DeviceInput, *DeviceOutput;
    cudaMalloc(&DeviceInput, N * sizeof(u32));
    cudaMalloc(&DeviceOutput, N * sizeof(u32));

    cudaMemcpy(DeviceInput, HostInput, N * sizeof(u32), cudaMemcpyHostToDevice);

    ReverseArray<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK, SHARED_DATA_SIZE>>>(DeviceInput, DeviceOutput, N);

    cudaMemcpy(HostOutput, DeviceOutput, N * sizeof(u32), cudaMemcpyDeviceToHost);

    // Print first and last few elements to verify
    printf("Input (first 8):  ");
    for (u32 I = 0; I < 8; I++)
        printf("%d ", HostInput[I]);
    printf("... ");
    for (u32 I = N - 8; I < N; I++)
        printf("%d ", HostInput[I]);

    printf("\nOutput (first 8): ");
    for (u32 I = 0; I < 8; I++)
        printf("%d ", HostOutput[I]);
    printf("... ");
    for (u32 I = N - 8; I < N; I++)
        printf("%d ", HostOutput[I]);
    printf("\n");

    // Verify correctness
    bool Correct = true;
    for (u32 I = 0; I < N; I++)
    {
        if (HostOutput[I] != N - 1 - I)
        {
            Correct = false;
            break;
        }
    }

    printf("\nGlobal reverse: %s\n", Correct ? "CORRECT" : "INCORRECT");

    // Cleanup
    free(HostInput);
    free(HostOutput);
    cudaFree(DeviceInput);
    cudaFree(DeviceOutput);

    printf("DAY_06: CUDA SUCCESS");

    return 0;
}
