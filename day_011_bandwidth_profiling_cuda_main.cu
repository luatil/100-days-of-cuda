#include <stdio.h>

#include "day_001_macros.h"
#include "day_001_vector_add_kernel.cu"

int main(int ArgumentCount, char **Arguments)
{
    u32 ExitCode = 0;

    if (ArgumentCount == 2)
    {
        u32 MaxPow = 0;
        sscanf(Arguments[1], "%d", &MaxPow);
        MaxPow = (MaxPow > 28) ? 28 : MaxPow;
        u32 MaxValue = 1 << MaxPow;

        for (u32 N = 1024; N <= MaxValue; N *= 2)
        {
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
                u32 ThreadsPerBlock = 512; // Tuning showed that 512 is a good value
                u32 BlocksPerGrid = (N + ThreadsPerBlock - 1) / ThreadsPerBlock;

                cudaEvent_t Start, Stop;
                cudaEventCreate(&Start);
                cudaEventCreate(&Stop);

                cudaEventRecord(Start);

                AddKernel<<<BlocksPerGrid, ThreadsPerBlock>>>(DeviceA, DeviceB, DeviceC, N);

                cudaEventRecord(Stop);
                cudaEventSynchronize(Stop);

                f32 Milliseconds = 0;
                cudaEventElapsedTime(&Milliseconds, Start, Stop);

                u32 BytesRead = 2.0f * SizeInBytes;
                u32 BytesWrote = 1.0f * SizeInBytes;
                f32 BytesProcessed = BytesRead + BytesWrote;
                f32 Bandwidth = (BytesProcessed / Milliseconds) * 1000.0f;
                f32 FlopsPerByte = 1.0f / (3.0f * 4); // 1 add vs (1 write + 2 read) - each f32 has 4 bytes

                fprintf(stdout, "N: %d\n", N);
                fprintf(stdout, "BlocksPerGrid: %d\n", BlocksPerGrid);
                fprintf(stdout, "ThreadsPerBlock: %d\n", ThreadsPerBlock);
                fprintf(stdout, "Kernel execution time: %f ms\n", Milliseconds);
                fprintf(stdout, "Bytes Processed: %.2f Mb\n", BytesProcessed / (1024 * 1024));
                fprintf(stdout, "Bandwidth: %.4f Gb/s\n", Bandwidth / (1024 * 1024 * 1024));
                fprintf(stdout, "Compute Throughput: %.4f GFLOPS/s\n", FlopsPerByte * Bandwidth / (1024 * 1024 * 1024));
                fprintf(stdout, "-------------------------\n");

                cudaEventDestroy(Start);
                cudaEventDestroy(Stop);
            }

            // Memcpy Device To Host
            {
                cudaMemcpy(HostC, DeviceC, SizeInBytes, cudaMemcpyDeviceToHost);
            }

            free(HostA);
            free(HostB);
            free(HostC);

            cudaFree(&DeviceA);
            cudaFree(&DeviceB);
            cudaFree(&DeviceC);
        }
    }
    else
    {
        fprintf(stderr, "Usage: %s [maximum number of vector elements]\n", Arguments[0]);
        ExitCode = 1;
    }

    return ExitCode;
}
