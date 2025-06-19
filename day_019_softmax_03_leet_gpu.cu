#include "day_019_common.h"
#include <cfloat>

typedef uint8_t u8;
typedef uint32_t u32;
typedef uint64_t u64;
typedef float f32;

#define Max(_a, _b) (_a > _b) ? _a : _b
#define Min(_a, _b) (_a < _b) ? _a : _b

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

    Shared[Tx] = -FLT_MIN;
    for (u32 I = 0; I < COARSE_FACTOR; I++)
    {
        if ((Tid + BLOCK_DIM * I) < N)
        {
            Shared[Tx] = Max(Shared[Tx], Input[Tid + BLOCK_DIM * I]);
        }
    }

    // printf("Tid = %d | Tx = %d | Shared[Tx] = %.5f\n", Tid, Tx, Shared[Tx]);
    for (u32 Stride = blockDim.x / 2; Stride >= 1; Stride /= 2)
    {
        __syncthreads();
        if (Tx < Stride)
        {
            Shared[Tx] = Max(Shared[Tx], Shared[Tx + Stride]);
        }
    }

    __syncthreads();
    // printf("Tid = %d | Tx = %d | Shared[Tx] = %.5f\n", Tid, Tx, Shared[Tx]);
    if (Tx == 0)
    {
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

    //  printf("Tid = %d | Tx = %d | Shared[Tx] = %.5f\n", Tid, Tx, Shared[Tx]);

    for (u32 Stride = blockDim.x / 2; Stride >= 1; Stride /= 2)
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
    u32 ThreadsPerBlock = Min(BLOCK_DIM, N);
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

#ifdef LEET_GPU
#include "solve.h"
#include <cfloat>
#include <cuda_runtime.h>
#include <stdint.h>

typedef uint8_t u8;
typedef uint32_t u32;
typedef uint64_t u64;
typedef float f32;

#define Max(_a, _b) (_a > _b) ? _a : _b
#define Min(_a, _b) (_a < _b) ? _a : _b

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

    Shared[Tx] = -FLT_MIN;
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

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float *input, float *output, int N)
{
    u32 ThreadsPerBlock = Min(BLOCK_DIM, N);
    u32 BlocksPerGrid = (N + (ThreadsPerBlock * COARSE_FACTOR) - 1) / (ThreadsPerBlock * COARSE_FACTOR);

    f32 *Device_GlobalMax, *Device_GlobalMaxSum;
    cudaMalloc(&Device_GlobalMax, sizeof(f32));
    cudaMalloc(&Device_GlobalMaxSum, sizeof(f32));

    SoftMax_Kernel_02_GlobalMax<<<BlocksPerGrid, ThreadsPerBlock>>>(input, Device_GlobalMax, N);
    SoftMax_Kernel_02_GlobalMaxSum<<<BlocksPerGrid, ThreadsPerBlock>>>(input, Device_GlobalMax, Device_GlobalMaxSum, N);

    // DbgCudaF32(Device_GlobalMaxSum);
    BlocksPerGrid = (N + (ThreadsPerBlock * 1) - 1) / (ThreadsPerBlock * 1);
    SoftMax_Kernel_02_Map<<<BlocksPerGrid, ThreadsPerBlock>>>(input, Device_GlobalMax, Device_GlobalMaxSum, output, N);
    cudaFree(Device_GlobalMax);
    cudaFree(Device_GlobalMax);
}

#endif
