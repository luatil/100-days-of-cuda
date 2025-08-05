#include <assert.h>
#include <stdio.h>

#define ASSERT(_Expr) assert(_Expr)
#define EPS 1e-8

__global__ void VectorSum(const float *A, const float *B, float *C, int N)
{
    int Tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (Tid < N)
    {
        C[Tid] = A[Tid] + B[Tid];
    }
}

__global__ void Linspace(float *X, float Init, float End, int N)
{
    int Tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (Tid < N)
    {
        X[Tid] = Init + ((End - Init) / (N - 1)) * Tid;
    }
}

__global__ void CheckResult(const float *A, const float *B, const float *C, int N)
{
    int Tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (Tid < N)
    {
        ASSERT(C[Tid] - (A[Tid] + B[Tid]) < EPS);
    }
}

__global__ void Print(float *X, int N)
{
    for (int I = 0; I < N; I++)
    {
        printf("%.4f ", X[I]);
    }
    printf("\n");
}

int main()
{
    int N = 1 << 18;
    float *A, *B, *C;
    cudaMalloc(&A, sizeof(float) * N);
    cudaMalloc(&B, sizeof(float) * N);
    cudaMalloc(&C, sizeof(float) * N);

    dim3 BlockDim(256);
    dim3 GridDim((N + 256 - 1) / 256);

    Linspace<<<GridDim, BlockDim>>>(A, 0.0, 100.0, N);
    Linspace<<<GridDim, BlockDim>>>(B, 0.0, 100.0, N);

    VectorSum<<<GridDim, BlockDim>>>(A, B, C, N);

    CheckResult<<<GridDim, BlockDim>>>(A, B, C, N);

    printf("First 5 elements of A:\n");
    Print<<<1, 1>>>(A, 5);
    cudaDeviceSynchronize();
    printf("First 5 elements of B:\n");
    Print<<<1, 1>>>(B, 5);
    cudaDeviceSynchronize();
    printf("First 5 elements of C:\n");
    Print<<<1, 1>>>(C, 5);
    cudaDeviceSynchronize();
}
