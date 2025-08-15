#include <stdint.h>
#include <stdio.h>

#include "day_003_vendor_libraries.h"

typedef uint8_t u8;
typedef uint32_t u32;

// NOTE(luatil): Assuming grayscale
struct image
{
    u8 *Data;
    float *DeviceData;
    u32 Width;
    u32 Height;
};

static void PrintCudaError(cudaError_t Error, const char *Operation)
{
    if (Error != cudaSuccess)
    {
        printf("CUDA Error in %s: %s (code %d)\n", Operation, cudaGetErrorString(Error), Error);
    }
}

// CUDA kernel to convert u8 to float
__global__ void ConvertU8ToFloat(u8 *input, float *output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output[idx] = (float)input[idx] / 255.0f; // Normalize to [0,1]
    }
}

// CUDA kernel to convert float back to u8
__global__ void ConvertFloatToU8(float *input, u8 *output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float val = input[idx] * 255.0f;
        // Clamp to [0, 255] range
        val = fmaxf(0.0f, fminf(255.0f, val));
        output[idx] = (u8)(val + 0.5f); // Round to nearest integer
    }
}

static image ReadImage(const char *Filename)
{
    image Result = {};

    // Load image using stb_image (grayscale)
    int x, y, n;
    unsigned char *data = stbi_load(Filename, &x, &y, &n, 1);
    if (!data)
    {
        printf("Error: Could not load image %s\n", Filename);
        return Result;
    }

    // Store host data and dimensions
    Result.Data = data;
    Result.Width = x;
    Result.Height = y;

    // Allocate device memory for u8 and float data
    u8 *deviceU8Data;
    int ImageSize = x * y;

    cudaError_t err1 = cudaMalloc((void **)&deviceU8Data, sizeof(u8) * ImageSize);
    cudaError_t err2 = cudaMalloc((void **)&Result.DeviceData, sizeof(float) * ImageSize);

    if (err1 != cudaSuccess || err2 != cudaSuccess)
    {
        PrintCudaError(err1, "Unable to malloc deviceU8Data");
        PrintCudaError(err2, "Unable to malloc device data");
        if (deviceU8Data)
            cudaFree(deviceU8Data);
        if (Result.DeviceData)
            cudaFree(Result.DeviceData);
        stbi_image_free(data);
        Result = {};
        return Result;
    }

    // Copy u8 data to device
    cudaError_t err3 = cudaMemcpy(deviceU8Data, data, sizeof(u8) * ImageSize, cudaMemcpyHostToDevice);
    if (err3 != cudaSuccess)
    {
        PrintCudaError(err3, "Error: CUDA memcpy failed\n");
        cudaFree(deviceU8Data);
        cudaFree(Result.DeviceData);
        stbi_image_free(data);
        Result = {};
        return Result;
    }

    // Launch kernel to convert u8 to float
    dim3 blockSize(256);
    dim3 gridSize((ImageSize + blockSize.x - 1) / blockSize.x);
    ConvertU8ToFloat<<<gridSize, blockSize>>>(deviceU8Data, Result.DeviceData, ImageSize);

    // Wait for kernel to complete and check for errors
    cudaError_t err4 = cudaDeviceSynchronize();
    if (err4 != cudaSuccess)
    {
        PrintCudaError(err4, "Kernel execution failed");
        cudaFree(deviceU8Data);
        cudaFree(Result.DeviceData);
        stbi_image_free(data);
        Result = {};
        return Result;
    }

    // Free temporary device u8 memory
    cudaFree(deviceU8Data);

    return Result;
}

static void WriteImage(image Image, const char *Filename)
{
    if (!Image.DeviceData || !Image.Data)
    {
        printf("Error: Invalid image data\n");
        return;
    }

    int ImageSize = Image.Width * Image.Height;

    // Allocate device memory for u8 output
    u8 *deviceU8Data;
    cudaError_t err1 = cudaMalloc((void **)&deviceU8Data, sizeof(u8) * ImageSize);
    if (err1 != cudaSuccess)
    {
        printf("Error: CUDA malloc failed in WriteImage\n");
        return;
    }

    // Launch kernel to convert float back to u8
    dim3 blockSize(256);
    dim3 gridSize((ImageSize + blockSize.x - 1) / blockSize.x);
    ConvertFloatToU8<<<gridSize, blockSize>>>(Image.DeviceData, deviceU8Data, ImageSize);

    // Wait for kernel to complete
    cudaError_t err2 = cudaDeviceSynchronize();
    if (err2 != cudaSuccess)
    {
        PrintCudaError(err2, "Kernel execution failed in WriteImage\n");
        cudaFree(deviceU8Data);
        return;
    }

    // Copy converted data back to host
    cudaError_t err3 = cudaMemcpy(Image.Data, deviceU8Data, sizeof(u8) * ImageSize, cudaMemcpyDeviceToHost);
    if (err3 != cudaSuccess)
    {
        printf("Error: CUDA memcpy failed in WriteImage\n");
        cudaFree(deviceU8Data);
        return;
    }

    int success = stbi_write_jpg(Filename, Image.Width, Image.Height, 1, Image.Data, 100);
    if (!success)
    {
        printf("Error: Could not write image %s\n", Filename);
        cudaFree(deviceU8Data);
        return;
    }
}

static void FreeImage(image Image)
{
    if (Image.Data)
    {
        stbi_image_free(Image.Data); // Use stbi_image_free instead of free
    }
    if (Image.DeviceData)
    {
        cudaFree(Image.DeviceData);
    }
}

__constant__ float RidgeKernel[3 * 3] = {
    -1.0f, -1.0f, -1.0f, -1.0f, 8.0f, -1.0f, -1.0f, -1.0f, -1.0f,
};

template <int BlockWidth = 16>
__global__ void RidgeConvolution2D(float *Input, float *Result, int InputWidth, int InputHeight, int KernelWidth = 3,
                                   int KernelHeight = 3)
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
                Sum += SharedInput[SharedIdx] * RidgeKernel[(M + HaloHeight) * KernelWidth + (N + HaloWidth)];
            }
        }

        Result[Row * InputWidth + Col] = Sum;
    }
}

static void Ridge(image Image)
{
    dim3 BlockSize(16, 16);
    dim3 GridSize((Image.Width + BlockSize.x - 1) / BlockSize.x, (Image.Height + BlockSize.y - 1) / BlockSize.y);

    int HaloWidth = 3 / 2;                // KernelWidth / 2
    int HaloHeight = 3 / 2;               // KernelHeight / 2
    int TileWidth = 16 + 2 * HaloWidth;   // BlockWidth + 2 * HaloWidth
    int TileHeight = 16 + 2 * HaloHeight; // BlockWidth + 2 * HaloHeight
    int SharedMemSize = TileWidth * TileHeight * sizeof(float);

    RidgeConvolution2D<<<GridSize, BlockSize, SharedMemSize>>>(Image.DeviceData, Image.DeviceData, Image.Width,
                                                               Image.Height);
}

static void EdgeDetect(image Image)
{
    Ridge(Image);
}

int main()
{
    image Image = ReadImage("data/nvidia_cuda_logo.jpg");
    EdgeDetect(Image);
    WriteImage(Image, "data/nvidia_edge_detect.jpg");
    FreeImage(Image);
}
