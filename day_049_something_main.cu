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
    float *DA, *DB, *DC;
    cudaMalloc(&DA, sizeof(float) * N);
    cudaMalloc(&DB, sizeof(float) * N);
    cudaMalloc(&DC, sizeof(float) * N);

    int BlockDim = 256;
    int GridDim = (N + BlockDim - 1) / BlockDim;

    FillValue<float><<<GridDim, BlockDim>>>(DA, 1.0f, N);
    FillValue<float><<<GridDim, BlockDim>>>(DB, 2.0f, N);

    VectorSumKernel<<<GridDim, BlockDim>>>(DA, DB, DC, N);

    PrintArray<10><<<1, 1>>>(DC, N);
    cudaDeviceSynchronize();
}
