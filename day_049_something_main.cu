#include <stdio.h>

#define MIN(_a, _b) ((_a < _b) ? _a : _b)

template <typename T> __global__ void FillValue(T *A, T Value, int N)
{
    int Tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (Tid < N)
    {
        A[Tid] = Value;
    }
}

template <int NumberOfElements> __global__ void PrintArray(float *A, int N)
{
#pragma unroll
    for (int I = 0; I < MIN(N, NumberOfElements); I++)
    {
        printf("%.3f\n", A[I]);
    }
}

__global__ void VectorSumKernel(float *A, float *B, float *C, int N)
{
    int Tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (Tid < N)
    {
        C[Tid] = A[Tid] + B[Tid];
    }
}

int main()
{
    int N = 1024 * 1024;
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeof(float) * N);
    cudaMalloc(&d_B, sizeof(float) * N);
    cudaMalloc(&d_C, sizeof(float) * N);

    int BlockDim = 256;
    int GridDim = (N + BlockDim - 1) / BlockDim;

    FillValue<float><<<GridDim, BlockDim>>>(d_A, 1.0f, N);
    FillValue<float><<<GridDim, BlockDim>>>(d_B, 2.0f, N);

    VectorSumKernel<<<GridDim, BlockDim>>>(d_A, d_B, d_C, N);

    PrintArray<10><<<1, 1>>>(d_C, N);
    cudaDeviceSynchronize();
}
