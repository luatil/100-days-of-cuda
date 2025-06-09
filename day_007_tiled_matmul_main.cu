/*
 * Day 07: Tiled MatMul Kernel
 *
 * Based on chapter 5 from PMPP.
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
#include "day_007_tiled_matmul_kernel.cu"

static f32 Eps = 1e-6;

int main()
{
    u32 N = 256;
    u32 M = 256;
    u32 SizeInBytes = sizeof(f32) * N * M;

    f32 *HostA = AllocateCPU(f32, N * M);
    f32 *HostB = AllocateCPU(f32, N * M);
    f32 *HostC = AllocateCPU(f32, N * M);

    for (u32 I = 0; I < (N * M); I++)
    {
        HostA[I] = 1.0f;
        HostB[I] = 2.0f;
    }

    f32 *DeviceA, *DeviceB, *DeviceC;
    cudaMalloc(&DeviceA, SizeInBytes);
    cudaMalloc(&DeviceB, SizeInBytes);
    cudaMalloc(&DeviceC, SizeInBytes);

    cudaMemcpy(DeviceA, HostA, SizeInBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(DeviceB, HostB, SizeInBytes, cudaMemcpyHostToDevice);

    dim3 ThreadsPerBlock(16, 16, 1);
    dim3 BlocksPerGrid((N + 16 - 1) / 16, (M + 16 - 1) / 16, 1);

    TiledMatmulKernel<<<BlocksPerGrid, ThreadsPerBlock>>>(DeviceA, DeviceB, DeviceC, N, N, N);

    cudaMemcpy(HostC, DeviceC, SizeInBytes, cudaMemcpyDeviceToHost);

    for (u32 I = 0; I < (N * M); I++)
    {
        f32 Exp = 2.0f * 256.0f;
        f32 Diff = HostC[I] - Exp;
        if (abs(Diff) > Eps)
        {
            printf("Cuda Kernel Failed | Pos: %d | Expected %f Got %f", I, Exp, HostC[I]);
            exit(1);
        }
    }

    printf("DAY_07: CUDA SUCCESS");
}
