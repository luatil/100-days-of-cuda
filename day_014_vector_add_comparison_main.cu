#include <stdio.h>

typedef unsigned int u32;
typedef float f32;

#include <stdio.h>

#include "day_001_macros.h"
#include "day_001_vector_add_kernel.cu"
#include "day_012_timed_cuda_block_002.cu"
#include "day_014_vector_add_float4_kernel.cu"

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

        {
            // Kernel Launch
            {
                TimeCudaBandwidth("Add Kernel Naive", 2 * SizeInBytes, SizeInBytes, 1.0f / (3.0f * 4.0f));

                int MinGridSize = 0, BlockSize = 0, GridSize = 0;

                cudaOccupancyMaxPotentialBlockSize(&MinGridSize, &BlockSize, AddKernel, 0, 0);

                fprintf(stdout, "MinGridSize=%d\n", MinGridSize);
                fprintf(stdout, "BlockSize=%d\n", BlockSize);

                GridSize = (N + BlockSize - 1) / BlockSize;

                fprintf(stdout, "MinGridSize=%d\n", MinGridSize);
                fprintf(stdout, "GridSize=%d\n", GridSize);
                fprintf(stdout, "Total Threads=%d\n", GridSize * BlockSize);

                AddKernel<<<GridSize, BlockSize>>>(DeviceA, DeviceB, DeviceC, N);
            }

            // Memcpy Device To Host
            {
                cudaMemcpy(HostC, DeviceC, SizeInBytes, cudaMemcpyDeviceToHost);
            }

            fprintf(stdout, "First element: %f\n", HostC[0]);
            fprintf(stdout, "Last element: %f\n", HostC[N - 1]);
        }

        {
            // Kernel Launch
            {
                TimeCudaBandwidth("Add Kernel Float4", 2 * SizeInBytes, SizeInBytes, 1.0f / (3.0f * 4.0f));

                int MinGridSize = 0, BlockSize = 0, GridSize = 0;

                cudaOccupancyMaxPotentialBlockSize(&MinGridSize, &BlockSize, AddVectorFloat4, 0, 0);

                GridSize = (N + (BlockSize * 2) - 1) / (BlockSize * 2);

                fprintf(stdout, "MinGridSize=%d\n", MinGridSize);
                fprintf(stdout, "GridSize=%d\n", GridSize);
                fprintf(stdout, "Total Threads=%d\n", GridSize * BlockSize);

                AddVectorFloat4<<<GridSize, BlockSize>>>(DeviceA, DeviceB, DeviceC, N);
            }

            // Memcpy Device To Host
            {
                cudaMemcpy(HostC, DeviceC, SizeInBytes, cudaMemcpyDeviceToHost);
            }

            fprintf(stdout, "First element: %f\n", HostC[0]);
            fprintf(stdout, "Last element: %f\n", HostC[N - 1]);
        }
    }
    else
    {
        fprintf(stderr, "Usage: %s [number of vector elements]\n", Arguments[0]);
        ExitCode = 1;
    }

    return ExitCode;
}
