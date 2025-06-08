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

#include "day_008_generate_random_matrix.cu"

#define AllocateCPU(_Type, _NumberOfElements) ((_Type *)malloc(sizeof(_Type) * (_NumberOfElements)))

int main()
{
    u32 N = 10;
    u32 M = 10;
    u32 SizeInBytes = sizeof(f32) * N * M;
    u64 Seed = 2345;

    f32 *HostMatrix = AllocateCPU(f32, N * M);

    f32 *DeviceMatrix;
    cudaMalloc(&DeviceMatrix, SizeInBytes);

    dim3 ThreadsPerBlock(32);
    dim3 BlocksPerGrid((N * M + 32 - 1) / 32);

    GenerateRandomMatrix<<<BlocksPerGrid, ThreadsPerBlock>>>(DeviceMatrix, N, M, Seed);

    cudaMemcpy(HostMatrix, DeviceMatrix, SizeInBytes, cudaMemcpyDeviceToHost);

    fprintf(stdout, "%d %d\n", N, M);
    for (u32 I = 0; I < N; I++)
    {
        for (u32 J = 0; J < M; J++)
        {
            fprintf(stdout, "%.6f ", HostMatrix[I * N + J]);
        }
        fprintf(stdout, "\n");
    }
}
