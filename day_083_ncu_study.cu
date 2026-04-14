#include <stdio.h>
#include <stdlib.h>

__global__ void AddKernel(float *A, float *B, float *C, int N)
{
    int Tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (Tid < N)
    {
        C[Tid] = A[Tid] + B[Tid];
    }
}

int main()
{
    const int N = 1024;
    float *A, *B, *C;
    A = (float *)malloc(N * sizeof(float));
    B = (float *)malloc(N * sizeof(float));
    C = (float *)malloc(N * sizeof(float));

    for (int I = 0; I < N; I++)
    {
        A[I] = 1.0f;
        B[I] = 2.0f;
        C[I] = 0.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeof(float) * N);
    cudaMalloc(&d_B, sizeof(float) * N);
    cudaMalloc(&d_C, sizeof(float) * N);

    cudaMemcpy(d_A, A, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(float) * N, cudaMemcpyHostToDevice);

    const int NumOfThreads = 32;
    const int NumOfBlocks = (N + NumOfThreads - 1) / NumOfThreads;
    AddKernel<<<NumOfBlocks, NumOfThreads>>>(d_A, d_B, d_C, N);

    cudaMemcpy(C, d_C, sizeof(float) * N, cudaMemcpyDeviceToHost);

    for (int I = 0; I < N; I++)
    {
        // printf("%.2f\n", C[I]);
    }
}
