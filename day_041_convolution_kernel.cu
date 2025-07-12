#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_SIZE 16
#define MAX_KERNEL_SIZE 9

__constant__ float d_kernel[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];

__global__ void ConvolutionKernel(unsigned char *input, unsigned char *output, int width, int height, int channels,
                                  int kernel_size, float scale, float bias)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Shared memory for input tile with padding
    __shared__ float shared_input[TILE_SIZE + MAX_KERNEL_SIZE - 1][TILE_SIZE + MAX_KERNEL_SIZE - 1];

    int padding = kernel_size / 2;
    int shared_width = TILE_SIZE + 2 * padding;
    int shared_height = TILE_SIZE + 2 * padding;

    // Load data into shared memory with boundary handling
    for (int dy = 0; dy < shared_height; dy += blockDim.y)
    {
        for (int dx = 0; dx < shared_width; dx += blockDim.x)
        {
            int shared_row = ty + dy;
            int shared_col = tx + dx;

            if (shared_row < shared_height && shared_col < shared_width)
            {
                int global_row = row - padding + dy;
                int global_col = col - padding + dx;

                // Handle boundary conditions with clamping
                global_row = max(0, min(height - 1, global_row));
                global_col = max(0, min(width - 1, global_col));

                shared_input[shared_row][shared_col] = (float)input[global_row * width + global_col] / 255.0f;
            }
        }
    }

    __syncthreads();

    if (row < height && col < width)
    {
        float result = 0.0f;

        // Apply convolution
        for (int ky = 0; ky < kernel_size; ky++)
        {
            for (int kx = 0; kx < kernel_size; kx++)
            {
                int shared_row = ty + padding + ky - padding;
                int shared_col = tx + padding + kx - padding;

                float kernel_val = d_kernel[ky * kernel_size + kx];
                float input_val = shared_input[shared_row][shared_col];

                result += kernel_val * input_val;
            }
        }

        // Apply scale and bias, then clamp to [0, 255]
        result = result * scale + bias;
        result = fmaxf(0.0f, fminf(255.0f, result));

        output[row * width + col] = (unsigned char)result;
    }
}

__global__ void ConvolutionKernelRGB(unsigned char *input, unsigned char *output, int width, int height, int channels,
                                     int kernel_size, float scale, float bias)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Shared memory for RGB channels
    __shared__ float shared_r[TILE_SIZE + MAX_KERNEL_SIZE - 1][TILE_SIZE + MAX_KERNEL_SIZE - 1];
    __shared__ float shared_g[TILE_SIZE + MAX_KERNEL_SIZE - 1][TILE_SIZE + MAX_KERNEL_SIZE - 1];
    __shared__ float shared_b[TILE_SIZE + MAX_KERNEL_SIZE - 1][TILE_SIZE + MAX_KERNEL_SIZE - 1];

    int padding = kernel_size / 2;
    int shared_width = TILE_SIZE + 2 * padding;
    int shared_height = TILE_SIZE + 2 * padding;

    // Load RGB data into shared memory
    for (int dy = 0; dy < shared_height; dy += blockDim.y)
    {
        for (int dx = 0; dx < shared_width; dx += blockDim.x)
        {
            int shared_row = ty + dy;
            int shared_col = tx + dx;

            if (shared_row < shared_height && shared_col < shared_width)
            {
                int global_row = row - padding + dy;
                int global_col = col - padding + dx;

                // Clamp coordinates
                global_row = max(0, min(height - 1, global_row));
                global_col = max(0, min(width - 1, global_col));

                int idx = (global_row * width + global_col) * channels;

                shared_r[shared_row][shared_col] = (float)input[idx] / 255.0f;
                shared_g[shared_row][shared_col] = (float)input[idx + 1] / 255.0f;
                shared_b[shared_row][shared_col] = (float)input[idx + 2] / 255.0f;
            }
        }
    }

    __syncthreads();

    if (row < height && col < width)
    {
        float result_r = 0.0f, result_g = 0.0f, result_b = 0.0f;

        // Apply convolution to each channel
        for (int ky = 0; ky < kernel_size; ky++)
        {
            for (int kx = 0; kx < kernel_size; kx++)
            {
                int shared_row = ty + padding + ky - padding;
                int shared_col = tx + padding + kx - padding;

                float kernel_val = d_kernel[ky * kernel_size + kx];

                result_r += kernel_val * shared_r[shared_row][shared_col];
                result_g += kernel_val * shared_g[shared_row][shared_col];
                result_b += kernel_val * shared_b[shared_row][shared_col];
            }
        }

        // Apply scale and bias, then clamp
        result_r = fmaxf(0.0f, fminf(255.0f, result_r * scale + bias));
        result_g = fmaxf(0.0f, fminf(255.0f, result_g * scale + bias));
        result_b = fmaxf(0.0f, fminf(255.0f, result_b * scale + bias));

        int idx = (row * width + col) * channels;
        output[idx] = (unsigned char)result_r;
        output[idx + 1] = (unsigned char)result_g;
        output[idx + 2] = (unsigned char)result_b;

        if (channels == 4)
        {
            output[idx + 3] = input[idx + 3]; // Copy alpha channel
        }
    }
}

void LaunchConvolution(unsigned char *d_input, unsigned char *d_output, int width, int height, int channels,
                       float *h_kernel, int kernel_size, float scale, float bias)
{
    // Copy kernel to constant memory
    cudaMemcpyToSymbol(d_kernel, h_kernel, kernel_size * kernel_size * sizeof(float));

    // Setup grid and block dimensions
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    if (channels == 1)
    {
        ConvolutionKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels, kernel_size, scale,
                                                   bias);
    }
    else
    {
        ConvolutionKernelRGB<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels, kernel_size, scale,
                                                      bias);
    }

    cudaDeviceSynchronize();
}
