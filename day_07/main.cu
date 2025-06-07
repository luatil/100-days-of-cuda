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

#define AllocateCPU(_Type, _NumberOfElements) ((_Type *)malloc(sizeof(_Type) * (_NumberOfElements)))

#ifndef DEBUG_ENABLED
#define DEBUG_ENABLED 1
#endif

#ifndef BLUR_SIZE
#define BLUR_SIZE 10
#endif

#if DEBUG_ENABLED
#define DbgU32(_Val) printf(#_Val "=%d\n", (_Val))
#define DbgS32(_Val) printf(#_Val "=%d\n", (_Val))
#define DbgF32(_Val) printf(#_Val "=%f\n", (_Val))
#else
#define DbgU32(_Val)
#define DbgS32(_Val)
#define DbgF32(_Val)
#endif

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
    
    ReverseArray<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK, SHARED_DATA_SIZE>>>(
        DeviceInput, DeviceOutput, N);
    
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
