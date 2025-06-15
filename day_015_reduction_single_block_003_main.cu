#include "day_015_common.h"

#define BLOCK_DIM 128

__global__ void ReductionKernel(f32 *Input, f32 *Output, u32 N)
{
    __shared__ f32 Shared[BLOCK_DIM];

    u32 Tid = threadIdx.x;

    Shared[Tid] = Input[Tid] + Input[Tid + BLOCK_DIM];

    for (u32 Stride = blockDim.x / 2; Stride >= 1; Stride /= 2)
    {
        __syncthreads();
        if (Tid < Stride)
        {
            Shared[Tid] += Shared[Tid + Stride];
        }
    }

    if (Tid == 0)
    {
        *Output = Shared[0];
    }
}

int main()
{
    const u32 N = 256;
    f32 *Input = AllocateCPU(f32, N);

    for (u32 I = 0; I < N; I++)
    {
        Input[I] = 1.0f;
    }

    f32 *Device_Input, *Device_Output;
    cudaMalloc(&Device_Input, sizeof(f32) * N);
    cudaMalloc(&Device_Output, sizeof(f32) * 1);

    cudaMemcpy(Device_Input, Input, sizeof(f32) * N, cudaMemcpyHostToDevice);

    u32 ThreadsPerBlock = N / 2;
    u32 BlocksPerGrid = 1;

    ReductionKernel<<<BlocksPerGrid, ThreadsPerBlock>>>(Device_Input, Device_Output, N);

    f32 Output;
    cudaMemcpy(&Output, Device_Output, sizeof(f32), cudaMemcpyDeviceToHost);

    fprintf(stdout, "%f\n", Output);
    Assert((N - Output) < 0.1);
}
