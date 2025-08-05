#include <cuda_runtime.h>
#include <stdio.h>

__global__ void Convolution2D(float *Input, float *Kernel, float *Result, int InputWidth, int InputHeight,
                              int KernelWidth, int KernelHeight)
{
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;

    if (Row < InputHeight && Col < InputWidth)
    {
        float Sum = 0.0f;

        for (int M = -KernelHeight / 2; M <= KernelHeight / 2; M++)
        {
            for (int N = -KernelWidth / 2; N <= KernelWidth / 2; N++)
            {
                int InputRow = Row + M;
                int InputCol = Col + N;

                if (InputRow >= 0 && InputRow < InputHeight && InputCol >= 0 && InputCol < InputWidth)
                {
                    Sum += Input[InputRow * InputWidth + InputCol] *
                           Kernel[(M + KernelHeight / 2) * KernelWidth + (N + KernelWidth / 2)];
                }
            }
        }

        Result[Row * InputWidth + Col] = Sum;
    }
}

int main()
{
    float Input[] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0,
                     14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0};

    float Result[sizeof(Input) / sizeof(Input[0])] = {};

    float Kernel[] = {0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625};

    int InputSize = sizeof(Input);
    int KernelSize = sizeof(Kernel);
    int ResultSize = sizeof(Result);

    float *DInput, *DKernel, *DResult;

    cudaMalloc(&DInput, InputSize);
    cudaMalloc(&DKernel, KernelSize);
    cudaMalloc(&DResult, ResultSize);

    for (int I = 0; I < 5; I++)
    {
        for (int J = 0; J < 5; J++)
        {
            printf("%8.3f ", Input[I * 5 + J]);
        }
        puts("");
    }
    puts("");
    for (int I = 0; I < 3; I++)
    {
        for (int J = 0; J < 3; J++)
        {
            printf("%8.3f ", Kernel[I * 3 + J]);
        }
        puts("");
    }

    int KernelHeight = 3;
    int KernelWidth = 3;
    int InputWidth = 5;
    int InputHeight = 5;

    cudaMemcpy(DInput, Input, InputSize, cudaMemcpyHostToDevice);
    cudaMemcpy(DKernel, Kernel, KernelSize, cudaMemcpyHostToDevice);

    dim3 BlockSize(16, 16);
    dim3 GridSize((InputWidth + BlockSize.x - 1) / BlockSize.x, (InputHeight + BlockSize.y - 1) / BlockSize.y);

    Convolution2D<<<GridSize, BlockSize>>>(DInput, DKernel, DResult, InputWidth, InputHeight, KernelWidth,
                                           KernelHeight);

    cudaMemcpy(Result, DResult, ResultSize, cudaMemcpyDeviceToHost);

    puts("Result");
    for (int I = 0; I < 5; I++)
    {
        for (int J = 0; J < 5; J++)
        {
            printf("%8.3f ", Result[I * 5 + J]);
        }
        puts("");
    }

    cudaFree(DInput);
    cudaFree(DKernel);
    cudaFree(DResult);

    return 0;
}
