#include <stdio.h>

typedef unsigned int u32;
typedef float f32;

#include <stdio.h>

#include "day_001_macros.h"
#include "day_001_vector_add_kernel.cu"
#include "day_012_timed_cuda_block_01.cu"

int main(int ArgumentCount, char **Arguments)
{
    u32 ExitCode = 0;

    if (ArgumentCount == 2)
    {
        u32 N = 0;
        sscanf(Arguments[1], "%d", &N);

        u32 SizeInBytes = sizeof(f32) * N;

        f32 *HostA = AllocateCPU(f32, N);
        f32 *HostB = AllocateCPU(f32, N);
        f32 *HostC = AllocateCPU(f32, N);

        for (u32 I = 0; I < N; I++)
        {
            HostA[I] = 1.0f * I;
            HostB[I] = 2.0f * I;
        }

        f32 *DeviceA, *DeviceB, *DeviceC;

        // Allocations
        {
            cudaMalloc(&DeviceA, SizeInBytes);
            cudaMalloc(&DeviceB, SizeInBytes);
            cudaMalloc(&DeviceC, SizeInBytes);
        }

        // Memcpy Host To Device
        {
            cudaMemcpy(DeviceA, HostA, SizeInBytes, cudaMemcpyHostToDevice);
            cudaMemcpy(DeviceB, HostB, SizeInBytes, cudaMemcpyHostToDevice);
        }

        // Kernel Launch
        {
            TimeCudaBlock("Add Kernel");
            u32 ThreadsPerBlock = 32;
            u32 BlocksPerGrid = (N + ThreadsPerBlock - 1) / ThreadsPerBlock;
            AddKernel<<<BlocksPerGrid, ThreadsPerBlock>>>(DeviceA, DeviceB, DeviceC, N);
        }

        // Memcpy Device To Host
        {
            cudaMemcpy(HostC, DeviceC, SizeInBytes, cudaMemcpyDeviceToHost);
        }
    }
    else
    {
        fprintf(stderr, "Usage: %s [number of vector elements]\n", Arguments[0]);
        ExitCode = 1;
    }

    return ExitCode;
}
