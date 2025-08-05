#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_SIZE 16
#define MAX_KERNEL_SIZE 9

__constant__ float DKernel[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];

__global__ void ConvolutionKernel(unsigned char *Input, unsigned char *Output, int Width, int Height, int Channels,
                                  int KernelSize, float Scale, float Bias)
{
    int Tx = threadIdx.x;
    int Ty = threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;

    // Shared memory for input tile with padding
    __shared__ float SharedInput[TILE_SIZE + MAX_KERNEL_SIZE - 1][TILE_SIZE + MAX_KERNEL_SIZE - 1];

    int Padding = KernelSize / 2;
    int SharedWidth = TILE_SIZE + 2 * Padding;
    int SharedHeight = TILE_SIZE + 2 * Padding;

    // Load data into shared memory with boundary handling
    for (int Dy = 0; Dy < SharedHeight; Dy += blockDim.y)
    {
        for (int Dx = 0; Dx < SharedWidth; Dx += blockDim.x)
        {
            int SharedRow = Ty + Dy;
            int SharedCol = Tx + Dx;

            if (SharedRow < SharedHeight && SharedCol < SharedWidth)
            {
                int GlobalRow = Row - Padding + Dy;
                int GlobalCol = Col - Padding + Dx;

                // Handle boundary conditions with clamping
                GlobalRow = max(0, min(Height - 1, GlobalRow));
                GlobalCol = max(0, min(Width - 1, GlobalCol));

                SharedInput[SharedRow][SharedCol] = (float)Input[GlobalRow * Width + GlobalCol] / 255.0f;
            }
        }
    }

    __syncthreads();

    if (Row < Height && Col < Width)
    {
        float Result = 0.0f;

        // Apply convolution
        for (int Ky = 0; Ky < KernelSize; Ky++)
        {
            for (int Kx = 0; Kx < KernelSize; Kx++)
            {
                int SharedRow = Ty + Padding + Ky - Padding;
                int SharedCol = Tx + Padding + Kx - Padding;

                float KernelVal = DKernel[Ky * KernelSize + Kx];
                float InputVal = SharedInput[SharedRow][SharedCol];

                Result += KernelVal * InputVal;
            }
        }

        // Apply scale and bias, then clamp to [0, 255]
        Result = Result * Scale + Bias;
        Result = fmaxf(0.0f, fminf(255.0f, Result));

        Output[Row * Width + Col] = (unsigned char)Result;
    }
}

__global__ void ConvolutionKernelRGB(unsigned char *Input, unsigned char *Output, int Width, int Height, int Channels,
                                     int KernelSize, float Scale, float Bias)
{
    int Tx = threadIdx.x;
    int Ty = threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;

    // Shared memory for RGB channels
    __shared__ float SharedR[TILE_SIZE + MAX_KERNEL_SIZE - 1][TILE_SIZE + MAX_KERNEL_SIZE - 1];
    __shared__ float SharedG[TILE_SIZE + MAX_KERNEL_SIZE - 1][TILE_SIZE + MAX_KERNEL_SIZE - 1];
    __shared__ float SharedB[TILE_SIZE + MAX_KERNEL_SIZE - 1][TILE_SIZE + MAX_KERNEL_SIZE - 1];

    int Padding = KernelSize / 2;
    int SharedWidth = TILE_SIZE + 2 * Padding;
    int SharedHeight = TILE_SIZE + 2 * Padding;

    // Load RGB data into shared memory
    for (int Dy = 0; Dy < SharedHeight; Dy += blockDim.y)
    {
        for (int Dx = 0; Dx < SharedWidth; Dx += blockDim.x)
        {
            int SharedRow = Ty + Dy;
            int SharedCol = Tx + Dx;

            if (SharedRow < SharedHeight && SharedCol < SharedWidth)
            {
                int GlobalRow = Row - Padding + Dy;
                int GlobalCol = Col - Padding + Dx;

                // Clamp coordinates
                GlobalRow = max(0, min(Height - 1, GlobalRow));
                GlobalCol = max(0, min(Width - 1, GlobalCol));

                int Idx = (GlobalRow * Width + GlobalCol) * Channels;

                SharedR[SharedRow][SharedCol] = (float)Input[Idx] / 255.0f;
                SharedG[SharedRow][SharedCol] = (float)Input[Idx + 1] / 255.0f;
                SharedB[SharedRow][SharedCol] = (float)Input[Idx + 2] / 255.0f;
            }
        }
    }

    __syncthreads();

    if (Row < Height && Col < Width)
    {
        float ResultR = 0.0f, ResultG = 0.0f, ResultB = 0.0f;

        // Apply convolution to each channel
        for (int Ky = 0; Ky < KernelSize; Ky++)
        {
            for (int Kx = 0; Kx < KernelSize; Kx++)
            {
                int SharedRow = Ty + Padding + Ky - Padding;
                int SharedCol = Tx + Padding + Kx - Padding;

                float KernelVal = DKernel[Ky * KernelSize + Kx];

                ResultR += KernelVal * SharedR[SharedRow][SharedCol];
                ResultG += KernelVal * SharedG[SharedRow][SharedCol];
                ResultB += KernelVal * SharedB[SharedRow][SharedCol];
            }
        }

        // Apply scale and bias, then clamp
        ResultR = fmaxf(0.0f, fminf(255.0f, ResultR * Scale + Bias));
        ResultG = fmaxf(0.0f, fminf(255.0f, ResultG * Scale + Bias));
        ResultB = fmaxf(0.0f, fminf(255.0f, ResultB * Scale + Bias));

        int Idx = (Row * Width + Col) * Channels;
        Output[Idx] = (unsigned char)ResultR;
        Output[Idx + 1] = (unsigned char)ResultG;
        Output[Idx + 2] = (unsigned char)ResultB;

        if (Channels == 4)
        {
            Output[Idx + 3] = Input[Idx + 3]; // Copy alpha channel
        }
    }
}

void LaunchConvolution(unsigned char *DInput, unsigned char *DOutput, int Width, int Height, int Channels,
                       float *HKernel, int KernelSize, float Scale, float Bias)
{
    // Copy kernel to constant memory
    cudaMemcpyToSymbol(DKernel, HKernel, KernelSize * KernelSize * sizeof(float));

    // Setup grid and block dimensions
    dim3 BlockSize(TILE_SIZE, TILE_SIZE);
    dim3 GridSize((Width + BlockSize.x - 1) / BlockSize.x, (Height + BlockSize.y - 1) / BlockSize.y);

    if (Channels == 1)
    {
        ConvolutionKernel<<<GridSize, BlockSize>>>(DInput, DOutput, Width, Height, Channels, KernelSize, Scale,
                                                   Bias);
    }
    else
    {
        ConvolutionKernelRGB<<<GridSize, BlockSize>>>(DInput, DOutput, Width, Height, Channels, KernelSize, Scale,
                                                      Bias);
    }

    cudaDeviceSynchronize();
}
