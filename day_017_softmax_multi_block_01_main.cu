#include "day_015_common.h"
#include <cfloat>
#include <math.h>
#include <stdlib.h>

#define BLOCK_SIZE 256

// Phase 1: Each block finds its local maximum
__global__ void SoftmaxFindMaxKernel(float *Input, float *GlobalMaxPerBlock, int SeqLen, int ElementsPerBlock)
{
    int BlockId = blockIdx.x;
    int Tid = threadIdx.x;
    int StartIdx = BlockId * ElementsPerBlock;
    int EndIdx = min(StartIdx + ElementsPerBlock, SeqLen);

    __shared__ float Sdata[BLOCK_SIZE];

    // Find local maximum for this block's chunk
    float ThreadMax = -FLT_MAX;
    for (int I = StartIdx + Tid; I < EndIdx; I += blockDim.x)
    {
        ThreadMax = fmaxf(ThreadMax, Input[I]);
    }
    Sdata[Tid] = ThreadMax;
    __syncthreads();

    // Block-level reduction to find local max
    for (int S = blockDim.x / 2; S > 0; S >>= 1)
    {
        if (Tid < S)
        {
            Sdata[Tid] = fmaxf(Sdata[Tid], Sdata[Tid + S]);
        }
        __syncthreads();
    }

    // Store this block's maximum
    if (Tid == 0)
    {
        GlobalMaxPerBlock[BlockId] = Sdata[0];
    }
}

// Phase 2: Reduce all local maxes to find global max
__global__ void SoftmaxReduceMaxKernel(float *GlobalMaxPerBlock, float *GlobalMax, int NumBlocks)
{
    int Tid = threadIdx.x;
    __shared__ float Sdata[BLOCK_SIZE];

    float ThreadMax = -FLT_MAX;
    for (int I = Tid; I < NumBlocks; I += blockDim.x)
    {
        ThreadMax = fmaxf(ThreadMax, GlobalMaxPerBlock[I]);
    }
    Sdata[Tid] = ThreadMax;
    __syncthreads();

    for (int S = blockDim.x / 2; S > 0; S >>= 1)
    {
        if (Tid < S)
        {
            Sdata[Tid] = fmaxf(Sdata[Tid], Sdata[Tid + S]);
        }
        __syncthreads();
    }

    if (Tid == 0)
    {
        *GlobalMax = Sdata[0];
    }
}

// Phase 3: Each block computes local sum of exp(x - global_max)
__global__ void SoftmaxComputeSumKernel(float *Input, float *GlobalSumPerBlock, float GlobalMax, int SeqLen,
                                           int ElementsPerBlock)
{
    int BlockId = blockIdx.x;
    int Tid = threadIdx.x;
    int StartIdx = BlockId * ElementsPerBlock;
    int EndIdx = min(StartIdx + ElementsPerBlock, SeqLen);

    __shared__ float Sdata[BLOCK_SIZE];

    // Compute local sum of exponentials
    float ThreadSum = 0.0f;
    for (int I = StartIdx + Tid; I < EndIdx; I += blockDim.x)
    {
        ThreadSum += expf(Input[I] - GlobalMax);
    }
    Sdata[Tid] = ThreadSum;
    __syncthreads();

    // Block-level reduction to find local sum
    for (int S = blockDim.x / 2; S > 0; S >>= 1)
    {
        if (Tid < S)
        {
            Sdata[Tid] += Sdata[Tid + S];
        }
        __syncthreads();
    }

    // Store this block's sum
    if (Tid == 0)
    {
        GlobalSumPerBlock[BlockId] = Sdata[0];
    }
}

// Phase 4: Reduce all local sums to find global sum
__global__ void SoftmaxReduceSumKernel(float *GlobalSumPerBlock, float *GlobalSum, int NumBlocks)
{
    int Tid = threadIdx.x;
    __shared__ float Sdata[BLOCK_SIZE];

    float ThreadSum = 0.0f;
    for (int I = Tid; I < NumBlocks; I += blockDim.x)
    {
        ThreadSum += GlobalSumPerBlock[I];
    }
    Sdata[Tid] = ThreadSum;
    __syncthreads();

    for (int S = blockDim.x / 2; S > 0; S >>= 1)
    {
        if (Tid < S)
        {
            Sdata[Tid] += Sdata[Tid + S];
        }
        __syncthreads();
    }

    if (Tid == 0)
    {
        *GlobalSum = Sdata[0];
    }
}

// Phase 5: Compute final softmax values
__global__ void SoftmaxFinalizeKernel(float *Input, float *Output, float GlobalMax, float GlobalSum, int SeqLen,
                                        int ElementsPerBlock)
{
    int BlockId = blockIdx.x;
    int Tid = threadIdx.x;
    int StartIdx = BlockId * ElementsPerBlock;
    int EndIdx = min(StartIdx + ElementsPerBlock, SeqLen);

    for (int I = StartIdx + Tid; I < EndIdx; I += blockDim.x)
    {
        Output[I] = expf(Input[I] - GlobalMax) / GlobalSum;
    }
}

void SoftmaxMultiBlock(float *Input, float *Output, int SeqLen)
{
    int ElementsPerBlock = BLOCK_SIZE * 4; // Each block handles 1024 elements
    int NumBlocks = (SeqLen + ElementsPerBlock - 1) / ElementsPerBlock;

    // Allocate intermediate arrays
    float *GlobalMaxPerBlock, *GlobalSumPerBlock;
    float *GlobalMax, *GlobalSum;

    cudaMalloc(&GlobalMaxPerBlock, sizeof(float) * NumBlocks);
    cudaMalloc(&GlobalSumPerBlock, sizeof(float) * NumBlocks);
    cudaMalloc(&GlobalMax, sizeof(float));
    cudaMalloc(&GlobalSum, sizeof(float));

    // Phase 1: Find local maxes
    SoftmaxFindMaxKernel<<<NumBlocks, BLOCK_SIZE>>>(Input, GlobalMaxPerBlock, SeqLen, ElementsPerBlock);
    cudaDeviceSynchronize();

    // Phase 2: Reduce to global max
    SoftmaxReduceMaxKernel<<<1, BLOCK_SIZE>>>(GlobalMaxPerBlock, GlobalMax, NumBlocks);
    cudaDeviceSynchronize();

    // Phase 3: Compute local sums
    SoftmaxComputeSumKernel<<<NumBlocks, BLOCK_SIZE>>>(Input, GlobalSumPerBlock, *GlobalMax, SeqLen,
                                                           ElementsPerBlock);
    cudaDeviceSynchronize();

    // Phase 4: Reduce to global sum
    SoftmaxReduceSumKernel<<<1, BLOCK_SIZE>>>(GlobalSumPerBlock, GlobalSum, NumBlocks);
    cudaDeviceSynchronize();

    // Phase 5: Compute final softmax
    SoftmaxFinalizeKernel<<<NumBlocks, BLOCK_SIZE>>>(Input, Output, *GlobalMax, *GlobalSum, SeqLen,
                                                        ElementsPerBlock);

    // Cleanup
    cudaFree(GlobalMaxPerBlock);
    cudaFree(GlobalSumPerBlock);
    cudaFree(GlobalMax);
    cudaFree(GlobalSum);
}

int main()
{
    const u32 SEQ_LEN = 10000; // Much larger than 1024!
    const u32 N = SEQ_LEN;

    f32 *Input = AllocateCPU(f32, N);
    f32 *Output = AllocateCPU(f32, N);

    // Initialize with test values
    for (u32 I = 0; I < N; I++)
    {
        Input[I] = (f32)(I % 10) + 1.0f;
    }

    f32 *DeviceInput, *DeviceOutput;
    cudaMalloc(&DeviceInput, sizeof(f32) * N);
    cudaMalloc(&DeviceOutput, sizeof(f32) * N);

    cudaMemcpy(DeviceInput, Input, sizeof(f32) * N, cudaMemcpyHostToDevice);

    SoftmaxMultiBlock(DeviceInput, DeviceOutput, SEQ_LEN);

    cudaMemcpy(Output, DeviceOutput, sizeof(f32) * N, cudaMemcpyDeviceToHost);

    // Verify softmax properties
    f32 Sum = 0.0f;
    for (u32 I = 0; I < SEQ_LEN; I++)
    {
        Sum += Output[I];
    }

    fprintf(stdout, "Sequence length: %d\n", SEQ_LEN);
    fprintf(stdout, "Softmax sum: %f (should be ~1.0)\n", Sum);
    fprintf(stdout, "First 5 values: ");
    for (u32 I = 0; I < 5; I++)
    {
        fprintf(stdout, "%.6f ", Output[I]);
    }
    fprintf(stdout, "\n");

    Assert(fabsf(Sum - 1.0f) < 0.001f);

    FreeCPU(Input);
    FreeCPU(Output);
    cudaFree(DeviceInput);
    cudaFree(DeviceOutput);
}
