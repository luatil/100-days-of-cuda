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

__global__ void LinSpaceKernel(float *X, float Start, float Step, int N)
{
    int Tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (Tid < N)
    {
        X[Tid] = Start + Tid * Step;
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

template<int BlockDim=256> __global__ void CopyKernel(float *V, float X, int N)
{
    int Tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (Tid < N)
    {
        V[Tid] = X;
    }
}

__global__ void PrintKernel(float *X, int N)
{
    for (int i = 0; i < N; i++)
    {
        printf("%f\n", X[i]);
    }
}

__device__ __host__ int Ceil(int Num, int Den)
{
    return (Num + Den - 1) / Den;
}

template <int Dim>
struct array
{
    float *m_Data;

    array(float X)
    {
        cudaMalloc(&m_Data, sizeof(float) * Dim);
        CopyKernel<<<Ceil(Dim, 256), 256>>>(m_Data, X, Dim);
    }


    template<typename... Args>
    array(Args... values)
    {
        static_assert(sizeof...(Args) == Dim, "Number of arguments must match array dimension");
        cudaMalloc(&m_Data, sizeof(float) * Dim);
        float host_data[Dim] = {static_cast<float>(values)...};
        cudaMemcpy(m_Data, host_data, sizeof(float) * Dim, cudaMemcpyHostToDevice);
    }

    array(array&& other) noexcept : m_Data(other.m_Data)
    {
        other.m_Data = nullptr;
    }

    float *data() const
    {
        return m_Data;
    }

    int size() const
    {
        return Dim;
    }

    array& operator=(array&& other) noexcept
    {
        if (this != &other)
        {
            cudaFree(m_Data);
            m_Data = other.m_Data;
            other.m_Data = nullptr;
        }
        return *this;
    }

    array(const array&) = delete;
    array& operator=(const array&) = delete;

    ~array()
    {
        if (m_Data)
        {
            cudaFree(m_Data);
        }
    }
};

template <int Dim>
array<Dim> operator+(const array<Dim>& A, const array<Dim>& B)
{
    array<Dim> Result(0.0f);
    AddKernel<<<Ceil(Dim, 256), 256>>>(A.m_Data, B.m_Data, Result.m_Data, Dim);
    return Result;
}

template <int Dim>
array<Dim> operator*(const array<Dim>& A, const array<Dim>& B)
{
    array<Dim> Result(0.0f);
    MultiplyKernel<<<Ceil(Dim, 256), 256>>>(A.m_Data, B.m_Data, Result.m_Data, Dim);
    return Result;
}

template <int Dim>
array<Dim> operator*(const array<Dim>& A, float X)
{
    array<Dim> Result(0.0f);
    MultiplyConstantKernel<<<Ceil(Dim, 256), 256>>>(A.m_Data, Result.m_Data, X, Dim);
    return Result;
}

template <int Dim>
array<1> Sum(const array<Dim>& A)
{
    array<1> Result(0.0f);
    SumKernel<<<Ceil(Dim, 256), 256>>>(A.m_Data, Result.m_Data, Dim);
    return Result;
}

template <int Dim>
void Print(const array<Dim>& Array)
{
    PrintKernel<<<1, 1>>>(Array.m_Data, Dim);
}

template <int Dim>
array<Dim> LinSpace(float Start, float Step)
{
    array<Dim> X(0.0f);
    LinSpaceKernel<<<Ceil(Dim, 256), 256>>>(X.m_Data, Start, Step, Dim);
    return X;
}
