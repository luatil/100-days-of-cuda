#include <cuda_runtime.h>
#include <stdio.h>

__constant__ float ConstKernel[21 * 21];

template <int BlockWidth = 16>
__global__ void Convolution2D(const float *Input, float *Result, int InputWidth, int InputHeight, int KernelWidth,
                              int KernelHeight)
{
    int Tx = threadIdx.x;
    int Ty = threadIdx.y;
    int Col = blockIdx.x * blockDim.x + Tx;
    int Row = blockIdx.y * blockDim.y + Ty;

    int HaloWidth = KernelWidth / 2;
    int HaloHeight = KernelHeight / 2;
    int TileWidth = BlockWidth + 2 * HaloWidth;
    int TileHeight = BlockWidth + 2 * HaloHeight;

    extern __shared__ float SharedInput[];

    // int SharedRow = Ty - HaloHeight;
    // int SharedCol = Tx - HaloWidth;

    for (int I = Ty; I < TileHeight; I += blockDim.y)
    {
        for (int J = Tx; J < TileWidth; J += blockDim.x)
        {
            int InputRow = blockIdx.y * blockDim.y + I - HaloHeight;
            int InputCol = blockIdx.x * blockDim.x + J - HaloWidth;

            if (InputRow >= 0 && InputRow < InputHeight && InputCol >= 0 && InputCol < InputWidth)
            {
                SharedInput[I * TileWidth + J] = Input[InputRow * InputWidth + InputCol];
            }
            else
            {
                SharedInput[I * TileWidth + J] = 0.0f;
            }
        }
    }

    __syncthreads();

    if (Row < InputHeight && Col < InputWidth)
    {
        float Sum = 0.0f;

        for (int M = -HaloHeight; M <= HaloHeight; M++)
        {
#pragma unroll
            for (int N = -HaloWidth; N <= HaloWidth; N++)
            {
                int SharedIdx = (Ty + HaloHeight + M) * TileWidth + (Tx + HaloWidth + N);
                Sum += SharedInput[SharedIdx] * ConstKernel[(M + HaloHeight) * KernelWidth + (N + HaloWidth)];
            }
        }

        Result[Row * InputWidth + Col] = Sum;
    }
}

// input, kernel, output are device pointers
extern "C" void solve(const float *Input, const float *Kernel, float *Output, int InputRows, int InputCols,
                      int KernelRows, int KernelCols)
{
    cudaMemcpyToSymbol(ConstKernel, Kernel, KernelRows * KernelCols * sizeof(float));

    dim3 BlockSize(16, 16);
    dim3 GridSize((InputRows + BlockSize.x - 1) / BlockSize.x, (InputCols + BlockSize.y - 1) / BlockSize.y);

    int TileWidth = 16 + KernelCols - 1;
    int TileHeight = 16 + KernelRows - 1;
    int SharedMemSize = TileWidth * TileHeight * sizeof(float);

    Convolution2D<<<GridSize, BlockSize, SharedMemSize>>>(Input, Output, InputRows, InputCols, KernelRows, KernelCols);
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
        printf("TEST PASSED: All convolution results match expected values\n");
    }
    else
    {
        printf("TEST FAILED: Some results do not match expected values\n");
    }

    cudaFree(DInput);
    cudaFree(DKernel);
    cudaFree(DResult);

    return 0;
}
#endif
