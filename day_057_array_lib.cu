#include <stdio.h>

__global__ void AddKernel(float *A, float *B, float *C, int N)
{
    int Tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (Tid < N)
    {
        C[Tid] = A[Tid] + B[Tid];
    }
}

__global__ void MultiplyKernel(float *A, float *B, float *C, int N)
{
    int Tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (Tid < N)
    {
        C[Tid] = A[Tid] * B[Tid];
    }
}

__global__ void MultiplyConstantKernel(float *A, float *B, float X, int N)
{
    int Tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (Tid < N)
    {
        B[Tid] = A[Tid] * X;
    }
}

__global__ void LinSpace(float *X, float Start, float End, int N)
{
    int Tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (Tid < N)
    {
        X[Tid] = Start + Tid * ((End - Start) / (N - 1));
    }
}

template <int BlockDim = 256> __global__ void SumKernel(float *X, float *Result, int N)
{
    __shared__ float Shared[BlockDim];

    int Tid = blockIdx.x * blockDim.x + threadIdx.x;
    int Tx = threadIdx.x;

    Shared[Tx] = Tid < N ? X[Tid] : 0.0f;
    __syncthreads();

    for (int Stride = BlockDim / 2; Stride > 0; Stride /= 2)
    {
        if (Tx < Stride)
        {
            Shared[Tx] += Shared[Tx + Stride];
        }
        __syncthreads();
    }

    if (Tx == 0)
    {
        atomicAdd(Result, Shared[0]);
    }
}

template <int BlockDim = 256> __global__ void CopyKernel(float *V, float X, int N)
{
    int Tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (Tid < N)
    {
        V[Tid] = X;
    }
}

__global__ void PrintKernel(float *X, int N)
{
    for (int I = 0; I < N; I++)
    {
        printf("%f\n", X[I]);
    }
}

__device__ __host__ int Ceil(int Num, int Den)
{
    return (Num + Den - 1) / Den;
}

template <int Dim> struct array
{
    float *MData;

    array(float X)
    {
        cudaMalloc(&MData, sizeof(float) * Dim);
        CopyKernel<<<Ceil(Dim, 256), 256>>>(MData, X, Dim);
    }

    array(array &&Other) noexcept : MData(Other.MData)
    {
        Other.MData = nullptr;
    }

    array &operator=(array &&Other) noexcept
    {
        if (this != &Other)
        {
            cudaFree(MData);
            MData = Other.MData;
            Other.MData = nullptr;
        }
        return *this;
    }

    array(const array &) = delete;
    array &operator=(const array &) = delete;

    ~array()
    {
        if (MData)
        {
            cudaFree(MData);
        }
    }
};

template <int Dim> array<Dim> operator+(const array<Dim> &A, const array<Dim> &B)
{
    array<Dim> Result(0.0f);
    AddKernel<<<Ceil(Dim, 256), 256>>>(A.MData, B.MData, Result.MData, Dim);
    return Result;
}

template <int Dim> array<Dim> operator*(const array<Dim> &A, const array<Dim> &B)
{
    array<Dim> Result(0.0f);
    MultiplyKernel<<<Ceil(Dim, 256), 256>>>(A.MData, B.MData, Result.MData, Dim);
    return Result;
}

template <int Dim> array<Dim> operator*(const array<Dim> &A, float X)
{
    array<Dim> Result(0.0f);
    MultiplyConstantKernel<<<Ceil(Dim, 256), 256>>>(A.MData, Result.MData, X, Dim);
    return Result;
}

template <int Dim> array<1> Sum(const array<Dim> &A)
{
    array<1> Result(0.0f);
    SumKernel<<<Ceil(Dim, 256), 256>>>(A.MData, Result.MData, Dim);
    return Result;
}

template <int Dim> void Print(const array<Dim> &Array)
{
    PrintKernel<<<1, 1>>>(Array.MData, Dim);
}

int main()
{
    const int N = 1024;

    array<N> Ones(1.0f);
    array<N> Twos(2.0f);
    array<N> Threes = Ones + Twos;
    array<N> Six = Threes * Twos;
    array<N> Twelve = Six * 2.0f;
    array<1> TwelveSum = Sum(Twelve);

    Print(TwelveSum);
    cudaDeviceSynchronize();
}
