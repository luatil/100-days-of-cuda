/*
 * Based on LeetGPU challenge
 *
 * # Understand the problem
 *
 * 1 ≤ N ≤ 500,000
 *
 * Since N > [maximun number of threads per block] we will need to use more than 1 block.
 *
 * For a numerical stable version we can use the following formula:
 *
 * e**(x-max(x)) / sum(e**(x-max(x)))
 *
 * Examples:
 *
 * Input: [1.0, 2.0, 3.0], N = 3 Output: [0.090, 0.244, 0.665] (approximately)
 * Input: [-10.0, -5.0, 0.0, 5.0, 10.0], N = 5 Output: [2.04e-09, 4.52e-07, 9.99e-01, 2.26e-02, 9.77e-01]
 * (approximately)
 *
 * # Plan a solution
 *
 * First I think I should create a simple CPU version to be able to verify the kernels.
 * For this we need to first generate some values to be able to test this version.
 *
 * After creating a CPU version to verify the results, we can play with different softmax kernels.
 *
 * The second most naive softmax kernel that would solve this problem would be to just use
 * 1 large block and do thread coarsening on the whole input.
 *
 * This should have abhorrent perfomance compared to the theorical maximun.
 * Considering that 500,000 / 1024 ~= 48.828125. This would be be
 * equivalent to a COARSE_FACTOR of 49 in the largest case.
 *
 * Interesting to see how this most naive softmax kernel compares to the naive
 * cpu version.
 *
 * After this we will prob. need to use more than 1 kernel to compute the final result.
 *
 * In our calculation there is we have a depency chain on global values, so I don't think
 * that we can do everything in a single kernel.
 *
 * [global maximun value] <- [global sum] <- [output value]
 *
 * So let's enumerate possible kernel solutions:
 *
 * 1. CPU Version 01
 * 2. GPU Version 01 (single block with thread coarsening)
 * 3. GPU Version 02 ( max reduction -> sum reduction -> map (lambda x: exp(x-max) / max_sum) )
 *
 *
 * # Implementing the solution:
 */

#include "day_015_common.h"
#include "day_017_make_random.h"
#include <cfloat>

#define Max(_a, _b) (_a > _b) ? _a : _b
#define Min(_a, _b) (_a < _b) ? _a : _b

static void CPU_SoftMax_01(const f32 *Input, f32 *Output, u32 N)
{
    f32 MaxValue = -FLT_MAX;
    for (u32 I = 0; I < N; I++)
    {
        MaxValue = Max(MaxValue, Input[I]);
    }

    f32 MaxSum = 0.0f;
    for (u32 I = 0; I < N; I++)
    {
        MaxSum += expf(Input[I] - MaxValue);
    }

    for (u32 I = 0; I < N; I++)
    {
        Output[I] = expf(Input[I] - MaxValue) / MaxSum;
    }
}

#define BLOCK_DIM 1024
__global__ void SoftMax_Kernel_01(const f32 *Input, f32 *Output, u32 N)
{
    int Tid = threadIdx.x;

    __shared__ f32 Shared[BLOCK_DIM];

    // Calculate the thread max with thread coarsening
    Shared[Tid] = -FLT_MAX;
    for (u32 I = Tid; I < N; I += BLOCK_DIM)
    {
        Shared[Tid] = Max(Shared[Tid], Input[I]);
    }
    __syncthreads();

    f32 GlobalMax = -FLT_MAX;
    for (u32 I = 0; I < BLOCK_DIM; I++)
    {
        GlobalMax = Max(GlobalMax, Shared[I]);
    }

    // Calculate the thread max with thread coarsening
    Shared[Tid] = 0.0f;
    for (u32 I = Tid; I < N; I += BLOCK_DIM)
    {
        Shared[Tid] += expf(Input[I] - GlobalMax);
    }
    __syncthreads();

    f32 GlobalMaxSum = 0.0f;
    for (u32 I = 0; I < BLOCK_DIM; I++)
    {
        GlobalMaxSum += Shared[I];
    }

    for (u32 I = Tid; I < N; I += BLOCK_DIM)
    {
        Output[I] = expf(Input[I] - GlobalMax) / GlobalMaxSum;
    }
}

static void GPU_SoftMax_01(const f32 *Device_Input, f32 *Device_Output, u32 N)
{
    u32 ThreadsPerBlock = Min(BLOCK_DIM, N);
    SoftMax_Kernel_01<<<1, ThreadsPerBlock>>>(Device_Input, Device_Output, N);
}
#undef BLOCK_DIM

int main()
{
    u32 N = 500000;
    // u64 Seed = 432432;
    f32 *Input = MakeSequentialF32(N);
    f32 *Output = AllocateCPU(f32, N);
    f32 *Reference_Output = AllocateCPU(f32, N);

    CPU_SoftMax_01(Input, Reference_Output, N);

    f32 ReferenceSum = 0.0f;
    for (u32 I = 0; I < N; I++)
    {
        ReferenceSum += Reference_Output[I];
        // printf("%f\n", Input[I]);
        // printf("%f\n", Reference_Output[I]);
    }

    fprintf(stdout, "Total Softmax: %.3f\n", ReferenceSum);

#if 1
    f32 *Device_Input, *Device_Output;
    cudaMalloc(&Device_Input, sizeof(f32) * N);
    cudaMalloc(&Device_Output, sizeof(f32) * N);

    cudaMemcpy(Device_Input, Input, sizeof(f32) * N, cudaMemcpyHostToDevice);

    GPU_SoftMax_01(Device_Input, Device_Output, N);

    cudaMemcpy(Output, Device_Output, sizeof(f32) * N, cudaMemcpyDeviceToHost);

    f32 OutputSum = 0.0f;
    for (u32 I = 0; I < N; I++)
    {
        // f32 Diff = Output[I] - Reference_Output[I];
        OutputSum += Output[I];
    }

    fprintf(stdout, "Softmax sum: %f (should be ~1.0)\n", OutputSum);
    fprintf(stdout, "Last 5 softmax values: ");
    for (u32 I = 0; I < 5 && I < N; I++)
    {
        fprintf(stdout, "%.6f ", Output[(N - 1) - I]);
    }
    fprintf(stdout, "\nLast 5 reference softmax values: ");
    for (u32 I = 0; I < 5 && I < N; I++)
    {
        fprintf(stdout, "%.6f ", Reference_Output[(N - 1) - I]);
    }
    fprintf(stdout, "\n");

    Assert(fabsf(OutputSum - 1.0f) < 0.001f);

    FreeCPU(Input);
    FreeCPU(Output);
    cudaFree(Device_Input);
    cudaFree(Device_Output);
#endif
}
