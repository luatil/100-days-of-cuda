/*
 * 1 <= N <= 100,000,000
 * 1 <= k <= N
 * input is 32 bit values
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef LEET_GPU
#include "solve.h"
#include <cuda_runtime.h>
#endif

__device__ void compare_swap(float *a, float *b, bool dir)
{
    if ((*a > *b) == dir)
    {
        float temp = *a;
        *a = *b;
        *b = temp;
    }
}

__global__ void simple_sort_kernel(float *data, int N)
{
    for (int i = 0; i < N - 1; i++)
    {
        for (int j = 0; j < N - 1 - i; j++)
        {
            if (data[j] < data[j + 1])
            {
                float temp = data[j];
                data[j] = data[j + 1];
                data[j + 1] = temp;
            }
        }
    }
}

// input, output are device pointers
void solve(const float *input, float *output, int N, int k)
{
    float *d_data;
    cudaError_t err = cudaMalloc(&d_data, N * sizeof(float));
    if (err != cudaSuccess)
    {
        printf("CUDA malloc failed: %s\n", cudaGetErrorString(err));
        return;
    }

    err = cudaMemcpy(d_data, input, N * sizeof(float), cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess)
    {
        printf("CUDA memcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return;
    }

    int block_size = 32;
    int grid_size = 1;

    simple_sort_kernel<<<grid_size, block_size>>>(d_data, N);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Sort kernel failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return;
    }

    cudaDeviceSynchronize();

    err = cudaMemcpy(output, d_data, k * sizeof(float), cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess)
    {
        printf("Output memcpy failed: %s\n", cudaGetErrorString(err));
    }

    cudaFree(d_data);
}

int main()
{
    const float h_input[] = {1.0f, 5.0f, 3.0f, 2.0f, 4.0f};
    int N = 5;
    int k = 3;

    const float expected_output[] = {5.0f, 4.0f, 3.0f};

    float *d_input;
    float *d_output;
    float *h_result = (float *)malloc(k * sizeof(float));

    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, k * sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    solve(d_input, d_output, N, k);

    cudaMemcpy(h_result, d_output, k * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Input: ");
    for (int i = 0; i < N; i++)
    {
        printf("%.1f ", h_input[i]);
    }
    printf("\n");

    printf("Top %d elements: ", k);
    for (int i = 0; i < k; i++)
    {
        printf("%.1f ", h_result[i]);
    }
    printf("\n");

    printf("Expected: ");
    for (int i = 0; i < k; i++)
    {
        printf("%.1f ", expected_output[i]);
    }
    printf("\n");

    bool correct = true;
    for (int i = 0; i < k; i++)
    {
        if (fabs(h_result[i] - expected_output[i]) > 1e-6)
        {
            correct = false;
            break;
        }
    }

    printf("Result: %s\n", correct ? "PASS" : "FAIL");

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_result);

    return 0;
}
