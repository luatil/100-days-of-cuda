#include <stdint.h>
#include <stdio.h>

typedef int32_t b32;
typedef uint32_t u32;
typedef uint64_t u64;
typedef float f32;

#include "day_001_macros.h"
#include "day_001_vector_add_kernel.cu"
#include "day_014_repetition_tester.cu"
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

        cudaMalloc(&DeviceA, SizeInBytes);
        cudaMalloc(&DeviceB, SizeInBytes);
        cudaMalloc(&DeviceC, SizeInBytes);

        cudaMemcpy(DeviceA, HostA, SizeInBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(DeviceB, HostB, SizeInBytes, cudaMemcpyHostToDevice);

        // Kernel Launch

        {
            int MinGridSize = 0, BlockSize = 0, GridSize = 0;

            cudaOccupancyMaxPotentialBlockSize(&MinGridSize, &BlockSize, AddKernel, 0, 0);
            GridSize = (N + BlockSize - 1) / BlockSize;

            fprintf(stdout, "MinGridSize=%d\n", MinGridSize);
            fprintf(stdout, "BlockSize=%d\n", BlockSize);
            fprintf(stdout, "GridSize=%d\n", GridSize);
            {
                cuda_repetition_tester Tester = {};
                StartTesting(&Tester, 2 * SizeInBytes, SizeInBytes, 1.0f / (3.0f * 4.0f), 2);

                while (IsTesting(&Tester))
                {
                    // TimeCudaBandwidth("Add Kernel", 2 * SizeInBytes, SizeInBytes, 1.0f / (3.0f * 4.0f));
                    BeginTime(&Tester);
                    AddKernel<<<GridSize, BlockSize>>>(DeviceA, DeviceB, DeviceC, N);
                    // cudaDeviceSyncronize(); This should be called by the repetition tester.
                    EndTime(&Tester);
                }
                PrintResults(&Tester, "Vector Add Naive");
            }

            // Memcpy Device To Host
            {
                cudaMemcpy(HostC, DeviceC, SizeInBytes, cudaMemcpyDeviceToHost);
                cudaMemset(DeviceC, 0, SizeInBytes);
            }

            fprintf(stdout, "First element: %f\n", HostC[0]);
            fprintf(stdout, "Last element: %f\n", HostC[N - 1]);
            fprintf(stdout, "Expected Last Element: %f\n", HostA[N - 1] + HostB[N - 1]);
        }

        {
            int MinGridSize = 0, BlockSize = 0, GridSize = 0;

            cudaOccupancyMaxPotentialBlockSize(&MinGridSize, &BlockSize, AddKernel, 0, 0);
            GridSize = (N + (BlockSize * 4) - 1) / (BlockSize * 4);

            fprintf(stdout, "MinGridSize=%d\n", MinGridSize);
            fprintf(stdout, "BlockSize=%d\n", BlockSize);
            fprintf(stdout, "GridSize=%d\n", GridSize);

            {
                cuda_repetition_tester Tester = {};
                StartTesting(&Tester, 2 * SizeInBytes, SizeInBytes, 1.0f / (3.0f * 4.0f), 2);

                while (IsTesting(&Tester))
                {
                    // TimeCudaBandwidth("Add Kernel", 2 * SizeInBytes, SizeInBytes, 1.0f / (3.0f * 4.0f));
                    BeginTime(&Tester);
                    AddVector_Float4<<<GridSize, BlockSize>>>(DeviceA, DeviceB, DeviceC, N);
                    // cudaDeviceSyncronize(); This should be called by the repetition tester.
                    EndTime(&Tester);
                }
                PrintResults(&Tester, "Vector Add Float4");
            }

            // Memcpy Device To Host
            {
                cudaMemcpy(HostC, DeviceC, SizeInBytes, cudaMemcpyDeviceToHost);
                cudaMemset(DeviceC, 0, SizeInBytes);
            }

            fprintf(stdout, "First element: %f\n", HostC[0]);
            fprintf(stdout, "Last element: %f\n", HostC[N - 1]);
            fprintf(stdout, "Expected Last Element: %f\n", HostA[N - 1] + HostB[N - 1]);
        }
    }
    else
    {
        fprintf(stderr, "Usage: %s [number of vector elements]\n", Arguments[0]);
        ExitCode = 1;
    }

    return ExitCode;
}
