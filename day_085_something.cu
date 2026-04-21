#include <cmath>
#include <stdio.h>

__global__ void Add(float *A, float *B, float *C, int N)
{
    int Tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (Tid < N)
    {
        C[Tid] = A[Tid] + B[Tid];
    }
}

__global__ void Set(float *V, float X, int N)
{
    int Tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (Tid < N)
    {
        V[Tid] = X;
    }
}

__global__ void MaxDifference(float *A, float *B, float *MaxDiff, int N)
{
    int Tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (Tid < N)
    {
        float Diff = abs(A[Tid] - B[Tid]);
        int DiffAsInt = __float_as_int(Diff);
        int *MaxDiffAsInt = (int *)MaxDiff;
        atomicMax(MaxDiffAsInt, DiffAsInt);
    }
}

int main(int argc, char *argv[])
{
    int N = 1024 * 1024; // 1 MB
    if (argc == 2)
    {
        N = atoi(argv[1]);
    }

    float *A, *B, *C, *Exp, *MaxDiff;
    cudaMalloc(&A, sizeof(float) * N);
    cudaMalloc(&B, sizeof(float) * N);
    cudaMalloc(&C, sizeof(float) * N);
    cudaMalloc(&Exp, sizeof(float) * N);
    cudaMalloc(&MaxDiff, sizeof(float) * 1);

    const int ThreadsPerBlock = 256;
    const int NumOfBlocks = (N + ThreadsPerBlock - 1) / ThreadsPerBlock;
    Set<<<NumOfBlocks, ThreadsPerBlock>>>(A, 1.0f, N);
    Set<<<NumOfBlocks, ThreadsPerBlock>>>(B, 5.0f, N);
    Set<<<NumOfBlocks, ThreadsPerBlock>>>(Exp, 4.0f, N);
    Add<<<NumOfBlocks, ThreadsPerBlock>>>(A, B, C, N);
    MaxDifference<<<NumOfBlocks, ThreadsPerBlock>>>(C, Exp, MaxDiff, N);

    float HostResult = 10.0f;
    cudaMemcpy(&HostResult, MaxDiff, sizeof(float) * 1, cudaMemcpyDeviceToHost);

    printf("MaxDiff = %f\n", HostResult);
}
