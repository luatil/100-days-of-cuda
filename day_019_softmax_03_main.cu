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

#include "day_019_common.h"
#include "day_019_make_random.h"
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
    Dbg(MaxValue);

    f32 MaxSum = 0.0f;
    for (u32 I = 0; I < N; I++)
    {
        MaxSum += expf(Input[I] - MaxValue);
    }
    Dbg(MaxSum);

    for (u32 I = 0; I < N; I++)
    {
        Output[I] = expf(Input[I] - MaxValue) / MaxSum;
    }
}

#define BLOCK_DIM 256
#define COARSE_FACTOR 2
__device__ float atomicMaxFloat(float *address, float val)
{
    int *address_as_int = (int *)address;
    int old = *address_as_int, assumed;
    do
    {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void SoftMax_Kernel_02_GlobalMax(const f32 *Input, f32 *GlobalMax, u32 N)
{
    __shared__ f32 Shared[BLOCK_DIM];

    u32 Segment = COARSE_FACTOR * blockDim.x * blockIdx.x;
    u32 Tid = Segment + threadIdx.x;
    u32 Tx = threadIdx.x;

    Shared[Tx] = -FLT_MAX;
    for (u32 I = 0; I < COARSE_FACTOR; I++)
    {
        if ((Tid + BLOCK_DIM * I) < N)
        {
            Shared[Tx] = Max(Shared[Tx], Input[Tid + BLOCK_DIM * I]);
        }
    }

    printf("Tid = %d | Tx = %d | Shared[Tx] = %.5f\n", Tid, Tx, Shared[Tx]);
    for (u32 Stride = (blockDim.x + 2 - 1) / 2; Stride >= 1; Stride /= 2)
    {
        __syncthreads();
        if (Tx < Stride)
        {
            Shared[Tx] = Max(Shared[Tx], Shared[Tx + Stride]);
        }
    }

    __syncthreads();
    if (Tx == 0)
    {
        printf("MAX: Tid = %d | Tx = %d | Shared[Tx] = %.5f\n", Tid, Tx, Shared[Tx]);
        atomicMaxFloat(GlobalMax, Shared[0]);
    }
}

__global__ void SoftMax_Kernel_02_GlobalMaxSum(const f32 *Input, const f32 *GlobalMax, f32 *GlobalMaxSum, u32 N)
{
    __shared__ f32 Shared[BLOCK_DIM];

    u32 Segment = COARSE_FACTOR * blockDim.x * blockIdx.x;
    u32 Tid = Segment + threadIdx.x;
    u32 Tx = threadIdx.x;

    f32 Sum = 0.0f;
    for (u32 I = 0; I < COARSE_FACTOR; I++)
    {
        if ((Tid + BLOCK_DIM * I) < N)
        {
            Sum += expf(Input[Tid + BLOCK_DIM * I] - *GlobalMax);
        }
    }
    Shared[Tx] = Sum;

    printf("Tid = %d | Tx = %d | Shared[Tx] = %.5f\n", Tid, Tx, Shared[Tx]);

    // for (u32 Stride = blockDim.x / 2; Stride >= 1; Stride /= 2)
    for (u32 Stride = (blockDim.x + 2 - 1) / 2; Stride >= 1; Stride /= 2)
    {
        __syncthreads();
        if (Tx < Stride)
        {
            Shared[Tx] += Shared[Tx + Stride];
        }
    }

    // NOTE(luatil): Shared[0] already has expfed value
    if (Tx == 0)
    {
        printf("SUM: Tid = %d | Tx = %d | Shared[Tx] = %.5f\n", Tid, Tx, Shared[Tx]);
        atomicAdd(GlobalMaxSum, Shared[0]);
    }
}

__global__ void SoftMax_Kernel_02_Map(const f32 *Input, const f32 *GlobalMax, const f32 *GlobalMaxSum, f32 *Output,
                                      u32 N)
{
    u32 Tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (Tid < N)
    {
        Output[Tid] = expf(Input[Tid] - *GlobalMax) / *GlobalMaxSum;
    }
}

static void GPU_SoftMax_02(const f32 *Device_Input, f32 *Device_Output, u32 N)
{
    u32 ThreadsPerBlock = BLOCK_DIM;
    u32 BlocksPerGrid = (N + (ThreadsPerBlock * COARSE_FACTOR) - 1) / (ThreadsPerBlock * COARSE_FACTOR);

    f32 *Device_GlobalMax, *Device_GlobalMaxSum;
    cudaMalloc(&Device_GlobalMax, sizeof(f32));
    cudaMalloc(&Device_GlobalMaxSum, sizeof(f32));

    SoftMax_Kernel_02_GlobalMax<<<BlocksPerGrid, ThreadsPerBlock>>>(Device_Input, Device_GlobalMax, N);
    // DbgCudaF32(Device_GlobalMax);
    SoftMax_Kernel_02_GlobalMaxSum<<<BlocksPerGrid, ThreadsPerBlock>>>(Device_Input, Device_GlobalMax,
                                                                       Device_GlobalMaxSum, N);

    // DbgCudaF32(Device_GlobalMaxSum);
    BlocksPerGrid = (N + (ThreadsPerBlock * 1) - 1) / (ThreadsPerBlock * 1);
    SoftMax_Kernel_02_Map<<<BlocksPerGrid, ThreadsPerBlock>>>(Device_Input, Device_GlobalMax, Device_GlobalMaxSum,
                                                              Device_Output, N);
    cudaFree(Device_GlobalMax);
    cudaFree(Device_GlobalMax);
}
#undef COARSE_FACTOR
#undef BLOCK_DIM

int main()
{
#if 0
    u32 N = 1000 * 50;
    // u32 N = 1024;
    // u64 Seed = 432432;
    f32 *Input = MakeSequentialF32(N);
#else
    u32 N = 4;
    f32 Input[] = {1000, -1.0, -2.0f, -3.0f};
#endif
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

    // NOTE(luatil): Also compare here with GPU_SoftMax_01
    GPU_SoftMax_02(Device_Input, Device_Output, N);

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

    // FreeCPU(Input);
    FreeCPU(Output);
    cudaFree(Device_Input);
    cudaFree(Device_Output);
#endif
}
