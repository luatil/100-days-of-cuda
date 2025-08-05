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

__device__ void CompareSwap(float *A, float *B, bool Dir)
{
    if ((*A > *B) == Dir)
    {
        float Temp = *A;
        *A = *B;
        *B = Temp;
    }
}

__global__ void SimpleSortKernel(float *Data, int N)
{
    for (int I = 0; I < N - 1; I++)
    {
        for (int J = 0; J < N - 1 - I; J++)
        {
            if (Data[J] < Data[J + 1])
            {
                float Temp = Data[J];
                Data[J] = Data[J + 1];
                Data[J + 1] = Temp;
            }
        }
    }
}

// input, output are device pointers
void Solve(const float *Input, float *Output, int N, int K)
{
    float *DData;
    cudaError_t Err = cudaMalloc(&DData, N * sizeof(float));
    if (Err != cudaSuccess)
    {
        printf("CUDA malloc failed: %s\n", cudaGetErrorString(Err));
        return;
    }

    Err = cudaMemcpy(DData, Input, N * sizeof(float), cudaMemcpyDeviceToDevice);
    if (Err != cudaSuccess)
    {
        printf("CUDA memcpy failed: %s\n", cudaGetErrorString(Err));
        cudaFree(DData);
        return;
    }

    int BlockSize = 32;
    int GridSize = 1;

    SimpleSortKernel<<<GridSize, BlockSize>>>(DData, N);
    Err = cudaGetLastError();
    if (Err != cudaSuccess)
    {
        printf("Sort kernel failed: %s\n", cudaGetErrorString(Err));
        cudaFree(DData);
        return;
    }

    cudaDeviceSynchronize();

    Err = cudaMemcpy(Output, DData, K * sizeof(float), cudaMemcpyDeviceToDevice);
    if (Err != cudaSuccess)
    {
        printf("Output memcpy failed: %s\n", cudaGetErrorString(Err));
    }

    cudaFree(DData);
}

int main()
{
    const float H_INPUT[] = {1.0f, 5.0f, 3.0f, 2.0f, 4.0f};
    int N = 5;
    int K = 3;

    const float EXPECTED_OUTPUT[] = {5.0f, 4.0f, 3.0f};

    float *DInput;
    float *DOutput;
    float *HResult = (float *)malloc(K * sizeof(float));

    cudaMalloc(&DInput, N * sizeof(float));
    cudaMalloc(&DOutput, K * sizeof(float));

    cudaMemcpy(DInput, H_INPUT, N * sizeof(float), cudaMemcpyHostToDevice);

    Solve(DInput, DOutput, N, K);

    cudaMemcpy(HResult, DOutput, K * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Input: ");
    for (int I = 0; I < N; I++)
    {
        printf("%.1f ", H_INPUT[I]);
    }
    printf("\n");

    printf("Top %d elements: ", K);
    for (int I = 0; I < K; I++)
    {
        printf("%.1f ", HResult[I]);
    }
    printf("\n");

    printf("Expected: ");
    for (int I = 0; I < K; I++)
    {
        printf("%.1f ", EXPECTED_OUTPUT[I]);
    }
    printf("\n");

    bool Correct = true;
    for (int I = 0; I < K; I++)
    {
        if (fabs(HResult[I] - EXPECTED_OUTPUT[I]) > 1e-6)
        {
            Correct = false;
            break;
        }
    }

    printf("Result: %s\n", Correct ? "PASS" : "FAIL");

    cudaFree(DInput);
    cudaFree(DOutput);
    free(HResult);

    return 0;
}
