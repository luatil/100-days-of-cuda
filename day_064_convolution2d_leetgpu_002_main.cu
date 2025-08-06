// #define LEET_GPU
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void Convolution2D(const float *Input, const float *Kernel, float *Result, int InputWidth, int InputHeight,
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

// input, kernel, output are device pointers
extern "C" void solve(const float *Input, const float *Kernel, float *Output, int InputRows, int InputCols,
                      int KernelRows, int KernelCols)
{
    dim3 BlockSize(16, 16);
    dim3 GridSize((InputRows + BlockSize.x - 1) / BlockSize.x, (InputCols + BlockSize.y - 1) / BlockSize.y);

    Convolution2D<<<GridSize, BlockSize>>>(Input, Kernel, Output, InputRows, InputCols, KernelRows, KernelCols);
}

#ifndef LEET_GPU
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

    solve(DInput, DKernel, DResult, InputWidth, InputHeight, KernelWidth, KernelHeight);
    cudaDeviceSynchronize();

    cudaMemcpy(Result, DResult, ResultSize, cudaMemcpyDeviceToHost);

    float Expected[] = {1.688,  2.750,  3.500,  4.250,  3.562,  4.750,  7.000,  8.000,  9.000,
                        7.250,  8.500,  12.000, 13.000, 14.000, 11.000, 12.250, 17.000, 18.000,
                        19.000, 14.750, 11.062, 15.250, 16.000, 16.750, 12.938};

    puts("Result");
    for (int I = 0; I < 5; I++)
    {
        for (int J = 0; J < 5; J++)
        {
            printf("%8.3f ", Result[I * 5 + J]);
        }
        puts("");
    }

    bool TestPassed = true;
    for (int I = 0; I < 25; I++)
    {
        if (abs(Result[I] - Expected[I]) > 0.001f)
        {
            TestPassed = false;
            printf("FAIL: Result[%d] = %f, Expected = %f\n", I, Result[I], Expected[I]);
        }
    }

    if (TestPassed)
    {
        printf("\nTEST PASSED: All convolution results match expected values\n");
    }
    else
    {
        printf("\nTEST FAILED: Some results do not match expected values\n");
    }

    cudaFree(DInput);
    cudaFree(DKernel);
    cudaFree(DResult);

    return 0;
}
#endif
