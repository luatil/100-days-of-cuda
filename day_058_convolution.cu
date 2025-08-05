#include "day_058_array_lib.cu"

__global__ void Convolution1DKernel(float *Input, float *Kernel, float *Output, int InputSize, int KernelSize,
                                    int OutputSize)
{
    int Tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (Tid < OutputSize)
    {
        float Sum = 0.0f;
        for (int I = 0; I < KernelSize; I++)
        {
            Sum += Input[Tid + I] * Kernel[I];
        }
        Output[Tid] = Sum;
    }
}

template <int InputDim, int KernelDim>
array<InputDim - KernelDim + 1> Convolution1D(const array<InputDim> &Input, const array<KernelDim> &Kernel)
{
    array<InputDim - KernelDim + 1> Result(0.0f);
    Convolution1DKernel<<<Ceil(InputDim - KernelDim + 1, 256), 256>>>(Input.data(), Kernel.data(), Result.data(),
                                                                      InputDim, KernelDim, InputDim - KernelDim + 1);
    return Result;
}

int main()
{
    auto Input = LinSpace<1024 * 1024>(1.0f, 1.0f);
    // Print(Input);

    array<4> Kernel{0.0f, 1.0f, 1.0f, 0.0f};
    // Print(Kernel);

    auto Result = Convolution1D<1024 * 1024, 4>(Input, Kernel);
    Print(Result);

    return 0;
}
