#include "day_014_repetition_tester.cu"
#include "day_015_common.h"
#include <stdlib.h>

#define BLOCK_DIM 256

#define COARSE_FACTOR 2 // Min Is 2
__global__ void ReductionKernelCoarseFactor2(f32 *Input, f32 *Output, u32 N)
{
    __shared__ f32 Shared[BLOCK_DIM];

    u32 Segment = COARSE_FACTOR * blockDim.x * blockIdx.x;
    u32 Tid = Segment + threadIdx.x;
    u32 Tx = threadIdx.x;

    for (u32 I = 0; I < COARSE_FACTOR; I++)
    {
        Shared[Tx] += Input[Tid + BLOCK_DIM * I];
    }

    for (u32 Stride = blockDim.x / 2; Stride >= 1; Stride /= 2)
    {
        __syncthreads();
        if (Tx < Stride)
        {
            Shared[Tx] += Shared[Tx + Stride];
        }
    }

    if (Tx == 0)
    {
        atomicAdd(Output, Shared[0]);
    }
}
#undef COARSE_FACTOR

#define COARSE_FACTOR 4 // Min Is 2
__global__ void ReductionKernelCoarseFactor4(f32 *Input, f32 *Output, u32 N)
{
    __shared__ f32 Shared[BLOCK_DIM];

    u32 Segment = COARSE_FACTOR * blockDim.x * blockIdx.x;
    u32 Tid = Segment + threadIdx.x;
    u32 Tx = threadIdx.x;

    for (u32 I = 0; I < COARSE_FACTOR; I++)
    {
        Shared[Tx] += Input[Tid + BLOCK_DIM * I];
    }

    for (u32 Stride = blockDim.x / 2; Stride >= 1; Stride /= 2)
    {
        __syncthreads();
        if (Tx < Stride)
        {
            Shared[Tx] += Shared[Tx + Stride];
        }
    }

    if (Tx == 0)
    {
        atomicAdd(Output, Shared[0]);
    }
}
#undef COARSE_FACTOR

#define COARSE_FACTOR 8 // Min Is 2
__global__ void ReductionKernelCoarseFactor8(f32 *Input, f32 *Output, u32 N)
{
    __shared__ f32 Shared[BLOCK_DIM];

    u32 Segment = COARSE_FACTOR * blockDim.x * blockIdx.x;
    u32 Tid = Segment + threadIdx.x;
    u32 Tx = threadIdx.x;

    for (u32 I = 0; I < COARSE_FACTOR; I++)
    {
        Shared[Tx] += Input[Tid + BLOCK_DIM * I];
    }

    for (u32 Stride = blockDim.x / 2; Stride >= 1; Stride /= 2)
    {
        __syncthreads();
        if (Tx < Stride)
        {
            Shared[Tx] += Shared[Tx + Stride];
        }
    }

    if (Tx == 0)
    {
        atomicAdd(Output, Shared[0]);
    }
}
#undef COARSE_FACTOR

#define COARSE_FACTOR 16 // Min Is 2
__global__ void ReductionKernelCoarseFactor16(f32 *Input, f32 *Output, u32 N)
{
    __shared__ f32 Shared[BLOCK_DIM];

    u32 Segment = COARSE_FACTOR * blockDim.x * blockIdx.x;
    u32 Tid = Segment + threadIdx.x;
    u32 Tx = threadIdx.x;

    for (u32 I = 0; I < COARSE_FACTOR; I++)
    {
        Shared[Tx] += Input[Tid + BLOCK_DIM * I];
    }

    for (u32 Stride = blockDim.x / 2; Stride >= 1; Stride /= 2)
    {
        __syncthreads();
        if (Tx < Stride)
        {
            Shared[Tx] += Shared[Tx + Stride];
        }
    }

    if (Tx == 0)
    {
        atomicAdd(Output, Shared[0]);
    }
}
#undef COARSE_FACTOR

int main(int argc, char **argv)
{
    u32 N = BLOCK_DIM * 256;

    if (argc > 1)
    {
        N = atoi(argv[1]);
        if (N == 0)
        {
            fprintf(stderr, "Invalid input size. Using default size.\n");
            N = BLOCK_DIM * 256;
        }
    }

    printf("Input size: %u elements\n", N);
    f32 *Input = AllocateCPU(f32, N);

    for (u32 I = 0; I < N; I++)
    {
        Input[I] = 1.0f;
    }

    f32 *DeviceInput, *DeviceOutput;
    cudaMalloc(&DeviceInput, sizeof(f32) * N);
    cudaMalloc(&DeviceOutput, sizeof(f32) * 1);
    cudaMemset(DeviceOutput, 0, sizeof(f32));

    cudaMemcpy(DeviceInput, Input, sizeof(f32) * N, cudaMemcpyHostToDevice);

    u32 ThreadsPerBlock = BLOCK_DIM;
    u32 SizeInBytes = sizeof(f32) * N;

    // Test Coarse Factor 2
    {
        u32 BlocksPerGrid = (N + (BLOCK_DIM * 2) - 1) / (BLOCK_DIM * 2);
        cuda_repetition_tester Tester = {};
        StartTesting(&Tester, SizeInBytes, sizeof(f32), 1.0f / 4.0f, 6);

        while (IsTesting(&Tester))
        {
            cudaMemset(DeviceOutput, 0, sizeof(f32));
            BeginTime(&Tester);
            ReductionKernelCoarseFactor2<<<BlocksPerGrid, ThreadsPerBlock>>>(DeviceInput, DeviceOutput, N);
            EndTime(&Tester);
        }
        PrintResults(&Tester, "Reduction Coarse Factor 2");
    }

    // Test Coarse Factor 4
    {
        u32 BlocksPerGrid = (N + (BLOCK_DIM * 4) - 1) / (BLOCK_DIM * 4);
        cuda_repetition_tester Tester = {};
        StartTesting(&Tester, SizeInBytes, sizeof(f32), 1.0f / 4.0f, 5);

        while (IsTesting(&Tester))
        {
            cudaMemset(DeviceOutput, 0, sizeof(f32));
            BeginTime(&Tester);
            ReductionKernelCoarseFactor4<<<BlocksPerGrid, ThreadsPerBlock>>>(DeviceInput, DeviceOutput, N);
            EndTime(&Tester);
        }
        PrintResults(&Tester, "Reduction Coarse Factor 4");
    }

    // Test Coarse Factor 8
    {
        u32 BlocksPerGrid = (N + (BLOCK_DIM * 8) - 1) / (BLOCK_DIM * 8);
        cuda_repetition_tester Tester = {};
        StartTesting(&Tester, SizeInBytes, sizeof(f32), 1.0f / 4.0f, 5);

        while (IsTesting(&Tester))
        {
            cudaMemset(DeviceOutput, 0, sizeof(f32));
            BeginTime(&Tester);
            ReductionKernelCoarseFactor8<<<BlocksPerGrid, ThreadsPerBlock>>>(DeviceInput, DeviceOutput, N);
            EndTime(&Tester);
        }
        PrintResults(&Tester, "Reduction Coarse Factor 8");
    }

    // Test Coarse Factor 16
    {
        u32 BlocksPerGrid = (N + (BLOCK_DIM * 16) - 1) / (BLOCK_DIM * 16);
        cuda_repetition_tester Tester = {};
        StartTesting(&Tester, SizeInBytes, sizeof(f32), 1.0f / 4.0f, 5);

        while (IsTesting(&Tester))
        {
            cudaMemset(DeviceOutput, 0, sizeof(f32));
            BeginTime(&Tester);
            ReductionKernelCoarseFactor16<<<BlocksPerGrid, ThreadsPerBlock>>>(DeviceInput, DeviceOutput, N);
            EndTime(&Tester);
        }
        PrintResults(&Tester, "Reduction Coarse Factor 16");
    }

    f32 Output;
    cudaMemcpy(&Output, DeviceOutput, sizeof(f32), cudaMemcpyDeviceToHost);

    fprintf(stdout, "%f\n", Output);
    Assert((N - Output) < 0.1);
}
