#include "day_015_common.h"

#define BLOCK_DIM 256
#define COARSE_FACTOR 4 // Min Is 2

__global__ void ReductionKernel(f32 *Input, f32 *Output, u32 N)
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

int main()
{
    const u32 N = BLOCK_DIM * 256;
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
    u32 BlocksPerGrid = (N + (BLOCK_DIM * COARSE_FACTOR) - 1) / (BLOCK_DIM * COARSE_FACTOR);

    ReductionKernel<<<BlocksPerGrid, ThreadsPerBlock>>>(DeviceInput, DeviceOutput, N);

    f32 Output;
    cudaMemcpy(&Output, DeviceOutput, sizeof(f32), cudaMemcpyDeviceToHost);

    fprintf(stdout, "Expected: %f\n", N * 1.0f);
    fprintf(stdout, "Output: %f\n", Output);
    Assert((N - Output) < 0.1);
}
