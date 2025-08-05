#include <cuda_runtime.h>
#include <stdio.h>

__global__ void Convolution2D(float *input, float *kernel, float *result, int inputWidth, int inputHeight,
                              int kernelWidth, int kernelHeight)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < inputHeight && col < inputWidth)
    {
        float sum = 0.0f;

        for (int m = -kernelHeight / 2; m <= kernelHeight / 2; m++)
        {
            for (int n = -kernelWidth / 2; n <= kernelWidth / 2; n++)
            {
                int inputRow = row + m;
                int inputCol = col + n;

                if (inputRow >= 0 && inputRow < inputHeight && inputCol >= 0 && inputCol < inputWidth)
                {
                    sum += input[inputRow * inputWidth + inputCol] *
                           kernel[(m + kernelHeight / 2) * kernelWidth + (n + kernelWidth / 2)];
                }
            }
        }

        result[row * inputWidth + col] = sum;
    }
}

int main()
{
    float Input[] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0,
                     14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0};

    float Result[sizeof(Input) / sizeof(Input[0])] = {};

    float Kernel[] = {0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625};

    int inputSize = sizeof(Input);
    int kernelSize = sizeof(Kernel);
    int resultSize = sizeof(Result);

    float *d_input, *d_kernel, *d_result;

    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_kernel, kernelSize);
    cudaMalloc(&d_result, resultSize);

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

    cudaMemcpy(d_input, Input, inputSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, Kernel, kernelSize, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((InputWidth + blockSize.x - 1) / blockSize.x, (InputHeight + blockSize.y - 1) / blockSize.y);

    Convolution2D<<<gridSize, blockSize>>>(d_input, d_kernel, d_result, InputWidth, InputHeight, KernelWidth,
                                           KernelHeight);

    cudaMemcpy(Result, d_result, resultSize, cudaMemcpyDeviceToHost);

    puts("Result");
    for (int I = 0; I < 5; I++)
    {
        for (int J = 0; J < 5; J++)
        {
            printf("%8.3f ", Result[I * 5 + J]);
        }
        puts("");
    }

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_result);

    return 0;
}
