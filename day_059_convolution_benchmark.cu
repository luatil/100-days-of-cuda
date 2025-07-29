#include <cuda_runtime.h>
#include <nvbench/nvbench.cuh>

#include "day_058_array_lib.cu"

__global__ void Convolution1DKernel(float *Input, float *Kernel, float *Output, int InputSize, int KernelSize, int OutputSize)
{
    int Tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (Tid < OutputSize)
    {
        float Sum = 0.0f;
        for (int i = 0; i < KernelSize; i++)
        {
            Sum += Input[Tid + i] * Kernel[i];
        }
        Output[Tid] = Sum;
    }
}

template <int InputDim, int KernelDim>
array<InputDim - KernelDim + 1> Convolution1D(const array<InputDim> &Input, const array<KernelDim> &Kernel)
{
    array<InputDim - KernelDim + 1> Result(0.0f);
    Convolution1DKernel<<<Ceil(InputDim - KernelDim + 1, 256), 256>>>(Input.data(), Kernel.data(), Result.data(), InputDim, KernelDim, InputDim - KernelDim + 1);
    return Result;
}

void test_benchmark(nvbench::state &state)
{
    constexpr int N = 1024*1024;
    array<N> Input(1.0f);
    array<4> Kernel{0.0f, 1.0f, 1.0f, 0.0f};

    // NOTE(luatil): Needs sync because of cudaSyncronize
    // that happens on the result destructor
    state.exec(nvbench::exec_tag::sync, [&](
        nvbench::launch &launch) {
        auto Result = Convolution1D<1024*1024, 4>(Input, Kernel);
    });
}

NVBENCH_BENCH(test_benchmark)
    .set_timeout(10);
