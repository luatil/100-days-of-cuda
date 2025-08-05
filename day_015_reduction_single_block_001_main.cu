#include "day_015_common.h"

__global__ void ReductionKernel(f32 *Input, f32 *Output, u32 N)
{
    u32 Tx = threadIdx.x;
    u32 Tid = Tx * 2;

    for (u32 Stride = 1; Stride <= blockDim.x; Stride *= 2)
    {
        if (Tx % Stride == 0)
        {
            Input[Tid] += Input[Tid + Stride];
        }
        __syncthreads();
    }

    if (Tx == 0)
    {
        *Output = Input[0];
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

    f32 *DeviceInput, *DeviceOutput;
    cudaMalloc(&DeviceInput, sizeof(f32) * N);
    cudaMalloc(&DeviceOutput, sizeof(f32) * 1);

    cudaMemcpy(DeviceInput, Input, sizeof(f32) * N, cudaMemcpyHostToDevice);

    u32 ThreadsPerBlock = N / 2;
    u32 BlocksPerGrid = 1;

    ReductionKernel<<<BlocksPerGrid, ThreadsPerBlock>>>(DeviceInput, DeviceOutput, N);

    f32 Output;
    cudaMemcpy(&Output, DeviceOutput, sizeof(f32), cudaMemcpyDeviceToHost);

    fprintf(stdout, "%f\n", Output);
    Assert((N - Output) < 0.1);
}
