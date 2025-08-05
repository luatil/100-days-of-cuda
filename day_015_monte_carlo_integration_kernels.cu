
// Input:  a = 0, b = 2, n_samples = 8
//         y_samples = [0.0625, 0.25, 0.5625, 1.0, 1.5625, 2.25, 3.0625, 4.0]
// Output: result = 3.1875
//
// Monte Carlo Integration
//
// ~ (b-a) * 1/n * sum for i in [1,n] y_i
#include "day_001_macros.h"

typedef float f32;
typedef unsigned int u32;

#define MIN(a, b) ((a < b) ? a : b)

typedef float f32;
typedef unsigned int u32;

typedef f32 monte_carlo_integration_function(const f32 *YSamples, f32 A, f32 B, u32 NumberOfSamples);

__host__ static f32 LaunchMonteCarloIntegrationCpu(const f32 *YSamples, f32 A, f32 B, u32 NumberOfSamples)
{
    f32 TotalSum = 0.0f;
    for (u32 I = 0; I < NumberOfSamples; I++)
    {
        TotalSum += YSamples[I];
    }

    f32 Result = (B - A) * TotalSum / NumberOfSamples;

    return Result;
}

#ifndef SHARED_MEM_WIDTH
#define SHARED_MEM_WIDTH 768
#endif

__global__ void MonteCarloIntegrationNaive(const f32 *YSamples, f32 *Result, f32 A, f32 B, u32 NumberOfSamples)
{
    __shared__ f32 SharedData[SHARED_MEM_WIDTH];

    u32 Tid = blockIdx.x * blockDim.x + threadIdx.x;
    u32 Tx = threadIdx.x;

    SharedData[Tx] = (Tid < NumberOfSamples) ? YSamples[Tid] : 0.0f;
    __syncthreads();

    if (Tx == 0)
    {
        for (u32 I = 0; I < MIN(SHARED_MEM_WIDTH, NumberOfSamples); I++)
        {
            Result[blockIdx.x] += SharedData[I];
        }
    }
}

// Y_Samples is a __host__ pointer
__host__ static f32 LaunchMonteCarloIntegrationNaive(const f32 *YSamples, f32 A, f32 B, u32 NumberOfSamples)
{
    // Copy Y_samples to gpu
    f32 *DeviceYSamples;
    cudaMalloc(&DeviceYSamples, sizeof(f32) * NumberOfSamples);

    cudaMemcpy(DeviceYSamples, YSamples, sizeof(f32) * NumberOfSamples, cudaMemcpyHostToDevice);

    u32 ThreadsPerGrid = MIN(SHARED_MEM_WIDTH, NumberOfSamples);
    u32 BlocksPerGrid = (NumberOfSamples + ThreadsPerGrid - 1) / ThreadsPerGrid;

    f32 *DeviceResultsPerBlock;
    cudaMalloc(&DeviceResultsPerBlock, sizeof(f32) * BlocksPerGrid);

    MonteCarloIntegrationNaive<<<BlocksPerGrid, ThreadsPerGrid>>>(YSamples, DeviceResultsPerBlock, A, B,
                                                                   NumberOfSamples);

    f32 *ResultsPerBlock = AllocateCPU(f32, BlocksPerGrid);

    cudaMemcpy(ResultsPerBlock, DeviceResultsPerBlock, sizeof(f32) * BlocksPerGrid, cudaMemcpyDeviceToHost);

    f32 TotalSum = 0.0f;

    for (u32 I = 0; I < BlocksPerGrid; I++)
    {
        TotalSum += ResultsPerBlock[I];
    }

    cudaFree(DeviceYSamples);
    cudaFree(DeviceResultsPerBlock);
    free(ResultsPerBlock);

    f32 Result = (B - A) * TotalSum / NumberOfSamples;

    return Result;
}

__global__ void MonteCarloIntegrationSimpleReduction(f32 *YSamples, f32 *Result, f32 A, f32 B, u32 NumberOfSamples)
{
    u32 Tid = threadIdx.x * 2;
    u32 Tx = threadIdx.x;

    for (u32 Stride = 1; Stride <= blockDim.x; Stride *= 2)
    {
        if (Tx % Stride == 0)
        {
            YSamples[Tid] += YSamples[Tid + Stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        *Result = YSamples[0];
    }
}

// Y_Samples is a __host__ pointer
__host__ static f32 LaunchMonteCarloIntegrationTreeReduction(const f32 *YSamples, f32 A, f32 B, u32 NumberOfSamples)
{
    // Copy Y_samples to gpu
    f32 *DeviceYSamples;
    cudaMalloc(&DeviceYSamples, sizeof(f32) * NumberOfSamples);

    cudaMemcpy(DeviceYSamples, YSamples, sizeof(f32) * NumberOfSamples, cudaMemcpyHostToDevice);

    u32 ThreadsPerGrid = MIN(SHARED_MEM_WIDTH, NumberOfSamples);
    u32 BlocksPerGrid = (NumberOfSamples + ThreadsPerGrid - 1) / ThreadsPerGrid;

    f32 *DeviceResultsPerBlock;
    cudaMalloc(&DeviceResultsPerBlock, sizeof(f32) * BlocksPerGrid);

    MonteCarloIntegrationNaive<<<BlocksPerGrid, ThreadsPerGrid>>>(YSamples, DeviceResultsPerBlock, A, B,
                                                                   NumberOfSamples);

    f32 *ResultsPerBlock = AllocateCPU(f32, BlocksPerGrid);

    cudaMemcpy(ResultsPerBlock, DeviceResultsPerBlock, sizeof(f32) * BlocksPerGrid, cudaMemcpyDeviceToHost);

    f32 TotalSum = 0.0f;

    for (u32 I = 0; I < BlocksPerGrid; I++)
    {
        TotalSum += ResultsPerBlock[I];
    }

    cudaFree(DeviceYSamples);
    cudaFree(DeviceResultsPerBlock);
    free(ResultsPerBlock);

    f32 Result = (B - A) * TotalSum / NumberOfSamples;

    return Result;
}
