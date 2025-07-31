#ifndef TENSOR_H
#define TENSOR_H

#include <cstddef>
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdio.h>

#define MAX_TENSOR_DIMS 8

template <int BlockDim = 256> __global__ void CopyKernel(float *V, float X, int N)
{
    int Tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (Tid < N)
    {
        V[Tid] = X;
    }
}

__global__ void AddKernel(float *A, float *B, float *C, int N)
{
    int Tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (Tid < N)
    {
        C[Tid] = A[Tid] + B[Tid];
    }
}

__global__ void SubtractKernel(float *A, float *B, float *C, int N)
{
    int Tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (Tid < N)
    {
        C[Tid] = A[Tid] - B[Tid];
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

__global__ void DivideKernel(float *A, float *B, float *C, int N)
{
    int Tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (Tid < N)
    {
        C[Tid] = A[Tid] / B[Tid];
    }
}

__global__ void AddScalarKernel(float *A, float *B, float X, int N)
{
    int Tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (Tid < N)
    {
        B[Tid] = A[Tid] + X;
    }
}

__global__ void SubtractScalarKernel(float *A, float *B, float X, int N)
{
    int Tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (Tid < N)
    {
        B[Tid] = A[Tid] - X;
    }
}

__global__ void MultiplyScalarKernel(float *A, float *B, float X, int N)
{
    int Tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (Tid < N)
    {
        B[Tid] = A[Tid] * X;
    }
}

__global__ void DivideScalarKernel(float *A, float *B, float X, int N)
{
    int Tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (Tid < N)
    {
        B[Tid] = A[Tid] / X;
    }
}

__global__ void MatMulKernel(float *A, float *B, float *C, int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
        {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__device__ __host__ int Ceil(int Num, int Den)
{
    return (Num + Den - 1) / Den;
}

struct tensor
{
    float *m_Data;
    int m_Dims[MAX_TENSOR_DIMS];
    int m_NDims;
    size_t m_Size;

    template <typename... Args> tensor(Args... values)
    {
        static_assert(sizeof...(Args) > 0, "Must provide at least one value");

        m_NDims = 1;
        m_Dims[0] = sizeof...(Args);
        m_Size = sizeof...(Args);

        cudaMalloc(&m_Data, m_Size * sizeof(float));

        float host_data[] = {static_cast<float>(values)...};
        cudaMemcpy(m_Data, host_data, m_Size * sizeof(float), cudaMemcpyHostToDevice);
    }

    tensor(tensor &&other) noexcept : m_Data(other.m_Data), m_NDims(other.m_NDims), m_Size(other.m_Size)
    {
        for (int i = 0; i < m_NDims; i++)
        {
            m_Dims[i] = other.m_Dims[i];
        }
        other.m_Data = nullptr;
    }

    tensor &operator=(tensor &&other) noexcept
    {
        if (this != &other)
        {
            cudaFree(m_Data);
            m_Data = other.m_Data;
            m_NDims = other.m_NDims;
            m_Size = other.m_Size;
            for (int i = 0; i < m_NDims; i++)
            {
                m_Dims[i] = other.m_Dims[i];
            }
            other.m_Data = nullptr;
        }
        return *this;
    }

    tensor(const tensor &) = delete;
    tensor &operator=(const tensor &) = delete;

    ~tensor()
    {
        if (m_Data)
        {
            cudaFree(m_Data);
        }
    }

    size_t Size()
    {
        size_t Result = 1;
        for (int I = 0; I < m_NDims; I++)
        {
            Result *= m_Dims[I];
        }
        return Result;
    }

    float *data() const
    {
        return m_Data;
    }

    void Print() const
    {
        if (!m_Data || m_Size == 0)
        {
            printf("[]\n");
            return;
        }

        float *host_data = (float *)malloc(m_Size * sizeof(float));
        cudaMemcpy(host_data, m_Data, m_Size * sizeof(float), cudaMemcpyDeviceToHost);

        if (m_NDims == 1)
        {
            printf("[");
            for (size_t i = 0; i < m_Size; i++)
            {
                printf("%.1f", host_data[i]);
                if (i < m_Size - 1)
                    printf(", ");
            }
            printf("]\n");
        }
        else if (m_NDims == 2)
        {
            printf("[");
            for (int i = 0; i < m_Dims[0]; i++)
            {
                printf("[");
                for (int j = 0; j < m_Dims[1]; j++)
                {
                    printf("%.1f", host_data[i * m_Dims[1] + j]);
                    if (j < m_Dims[1] - 1)
                        printf(", ");
                }
                printf("]");
                if (i < m_Dims[0] - 1)
                    printf(", ");
            }
            printf("]\n");
        }
        else
        {
            printf("Shape: (");
            for (int i = 0; i < m_NDims; i++)
            {
                printf("%d", m_Dims[i]);
                if (i < m_NDims - 1)
                    printf(", ");
            }
            printf(") Data: [");
            for (size_t i = 0; i < m_Size; i++)
            {
                printf("%.1f", host_data[i]);
                if (i < m_Size - 1)
                    printf(", ");
            }
            printf("]\n");
        }

        free(host_data);
    }

    template <typename... Args> tensor Reshape(Args... dims)
    {
        static_assert(sizeof...(Args) > 0, "Must provide at least one dimension");

        int new_dims[] = {dims...};
        int new_ndims = sizeof...(Args);

        size_t new_size = 1;
        for (int i = 0; i < new_ndims; i++)
        {
            new_size *= new_dims[i];
        }

        if (new_size != m_Size)
        {
            printf("Error: Cannot reshape tensor of size %zu to size %zu\n", m_Size, new_size);
            tensor empty;
            return empty;
        }

        tensor Result;
        Result.m_Data = m_Data;
        Result.m_NDims = new_ndims;
        Result.m_Size = m_Size;

        for (int i = 0; i < new_ndims; i++)
        {
            Result.m_Dims[i] = new_dims[i];
        }

        m_Data = nullptr;

        return Result;
    }

    tensor MatMul(const tensor &B) const
    {
        if (m_NDims != 2 || B.m_NDims != 2)
        {
            printf("Error: MatMul requires both tensors to be 2D. Got %dD and %dD\n", m_NDims, B.m_NDims);
            tensor empty;
            return empty;
        }

        int M = m_Dims[0];   // Rows of A
        int K = m_Dims[1];   // Cols of A / Rows of B
        int N = B.m_Dims[1]; // Cols of B

        if (m_Dims[1] != B.m_Dims[0])
        {
            printf("Error: MatMul incompatible shapes: (%d, %d) × (%d, %d)\n", m_Dims[0], m_Dims[1], B.m_Dims[0],
                   B.m_Dims[1]);
            tensor empty;
            return empty;
        }

        tensor result;
        result.m_NDims = 2;
        result.m_Dims[0] = M;
        result.m_Dims[1] = N;
        result.m_Size = M * N;

        cudaMalloc(&result.m_Data, result.m_Size * sizeof(float));

        dim3 blockSize(16, 16);
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);

        MatMulKernel<<<gridSize, blockSize>>>(m_Data, B.m_Data, result.m_Data, M, N, K);

        return result;
    }

    tensor &operator+=(const tensor &other)
    {
        if (!IsShapeCompatible(other))
        {
            printf("Error: Incompatible tensor shapes for += operation\n");
            return *this;
        }

        AddKernel<<<Ceil(m_Size, 256), 256>>>(m_Data, other.m_Data, m_Data, m_Size);
        return *this;
    }

    tensor &operator-=(const tensor &other)
    {
        if (!IsShapeCompatible(other))
        {
            printf("Error: Incompatible tensor shapes for -= operation\n");
            return *this;
        }

        SubtractKernel<<<Ceil(m_Size, 256), 256>>>(m_Data, other.m_Data, m_Data, m_Size);
        return *this;
    }

    tensor &operator*=(const tensor &other)
    {
        if (!IsShapeCompatible(other))
        {
            printf("Error: Incompatible tensor shapes for *= operation\n");
            return *this;
        }

        MultiplyKernel<<<Ceil(m_Size, 256), 256>>>(m_Data, other.m_Data, m_Data, m_Size);
        return *this;
    }

    tensor &operator/=(const tensor &other)
    {
        if (!IsShapeCompatible(other))
        {
            printf("Error: Incompatible tensor shapes for /= operation\n");
            return *this;
        }

        DivideKernel<<<Ceil(m_Size, 256), 256>>>(m_Data, other.m_Data, m_Data, m_Size);
        return *this;
    }

    tensor &operator+=(float scalar)
    {
        AddScalarKernel<<<Ceil(m_Size, 256), 256>>>(m_Data, m_Data, scalar, m_Size);
        return *this;
    }

    tensor &operator-=(float scalar)
    {
        SubtractScalarKernel<<<Ceil(m_Size, 256), 256>>>(m_Data, m_Data, scalar, m_Size);
        return *this;
    }

    tensor &operator*=(float scalar)
    {
        MultiplyScalarKernel<<<Ceil(m_Size, 256), 256>>>(m_Data, m_Data, scalar, m_Size);
        return *this;
    }

    tensor &operator/=(float scalar)
    {
        DivideScalarKernel<<<Ceil(m_Size, 256), 256>>>(m_Data, m_Data, scalar, m_Size);
        return *this;
    }

    tensor() : m_Data(nullptr), m_NDims(0), m_Size(0)
    {
    }

    bool IsShapeCompatible(const tensor &other) const
    {
        if (m_NDims != other.m_NDims || m_Size != other.m_Size)
            return false;

        for (int i = 0; i < m_NDims; i++)
        {
            if (m_Dims[i] != other.m_Dims[i])
                return false;
        }
        return true;
    }

  private:
};

tensor operator+(const tensor &A, const tensor &B)
{
    if (!A.IsShapeCompatible(B))
    {
        printf("Error: Incompatible tensor shapes for addition\n");
        tensor empty;
        return empty;
    }

    tensor Result;
    Result.m_NDims = A.m_NDims;
    Result.m_Size = A.m_Size;
    for (int i = 0; i < A.m_NDims; i++)
    {
        Result.m_Dims[i] = A.m_Dims[i];
    }

    cudaMalloc(&Result.m_Data, Result.m_Size * sizeof(float));
    AddKernel<<<Ceil(Result.m_Size, 256), 256>>>(A.m_Data, B.m_Data, Result.m_Data, Result.m_Size);

    return Result;
}

tensor operator-(const tensor &A, const tensor &B)
{
    if (!A.IsShapeCompatible(B))
    {
        printf("Error: Incompatible tensor shapes for subtraction\n");
        tensor empty;
        return empty;
    }

    tensor Result;
    Result.m_NDims = A.m_NDims;
    Result.m_Size = A.m_Size;
    for (int i = 0; i < A.m_NDims; i++)
    {
        Result.m_Dims[i] = A.m_Dims[i];
    }

    cudaMalloc(&Result.m_Data, Result.m_Size * sizeof(float));
    SubtractKernel<<<Ceil(Result.m_Size, 256), 256>>>(A.m_Data, B.m_Data, Result.m_Data, Result.m_Size);

    return Result;
}

tensor operator*(const tensor &A, const tensor &B)
{
    if (!A.IsShapeCompatible(B))
    {
        printf("Error: Incompatible tensor shapes for multiplication\n");
        tensor empty;
        return empty;
    }

    tensor Result;
    Result.m_NDims = A.m_NDims;
    Result.m_Size = A.m_Size;
    for (int i = 0; i < A.m_NDims; i++)
    {
        Result.m_Dims[i] = A.m_Dims[i];
    }

    cudaMalloc(&Result.m_Data, Result.m_Size * sizeof(float));
    MultiplyKernel<<<Ceil(Result.m_Size, 256), 256>>>(A.m_Data, B.m_Data, Result.m_Data, Result.m_Size);

    return Result;
}

tensor operator/(const tensor &A, const tensor &B)
{
    if (!A.IsShapeCompatible(B))
    {
        printf("Error: Incompatible tensor shapes for division\n");
        tensor empty;
        return empty;
    }

    tensor Result;
    Result.m_NDims = A.m_NDims;
    Result.m_Size = A.m_Size;
    for (int i = 0; i < A.m_NDims; i++)
    {
        Result.m_Dims[i] = A.m_Dims[i];
    }

    cudaMalloc(&Result.m_Data, Result.m_Size * sizeof(float));
    DivideKernel<<<Ceil(Result.m_Size, 256), 256>>>(A.m_Data, B.m_Data, Result.m_Data, Result.m_Size);

    return Result;
}

tensor operator+(const tensor &A, float scalar)
{
    tensor Result;
    Result.m_NDims = A.m_NDims;
    Result.m_Size = A.m_Size;
    for (int i = 0; i < A.m_NDims; i++)
    {
        Result.m_Dims[i] = A.m_Dims[i];
    }

    cudaMalloc(&Result.m_Data, Result.m_Size * sizeof(float));
    AddScalarKernel<<<Ceil(Result.m_Size, 256), 256>>>(A.m_Data, Result.m_Data, scalar, Result.m_Size);

    return Result;
}

tensor operator+(float scalar, const tensor &A)
{
    return A + scalar;
}

tensor operator-(const tensor &A, float scalar)
{
    tensor Result;
    Result.m_NDims = A.m_NDims;
    Result.m_Size = A.m_Size;
    for (int i = 0; i < A.m_NDims; i++)
    {
        Result.m_Dims[i] = A.m_Dims[i];
    }

    cudaMalloc(&Result.m_Data, Result.m_Size * sizeof(float));
    SubtractScalarKernel<<<Ceil(Result.m_Size, 256), 256>>>(A.m_Data, Result.m_Data, scalar, Result.m_Size);

    return Result;
}

tensor operator-(float scalar, const tensor &A)
{
    tensor Result;
    Result.m_NDims = A.m_NDims;
    Result.m_Size = A.m_Size;
    for (int i = 0; i < A.m_NDims; i++)
    {
        Result.m_Dims[i] = A.m_Dims[i];
    }

    cudaMalloc(&Result.m_Data, Result.m_Size * sizeof(float));
    AddScalarKernel<<<Ceil(Result.m_Size, 256), 256>>>(A.m_Data, Result.m_Data, -scalar, Result.m_Size);

    return Result;
}

tensor operator*(const tensor &A, float scalar)
{
    tensor Result;
    Result.m_NDims = A.m_NDims;
    Result.m_Size = A.m_Size;
    for (int i = 0; i < A.m_NDims; i++)
    {
        Result.m_Dims[i] = A.m_Dims[i];
    }

    cudaMalloc(&Result.m_Data, Result.m_Size * sizeof(float));
    MultiplyScalarKernel<<<Ceil(Result.m_Size, 256), 256>>>(A.m_Data, Result.m_Data, scalar, Result.m_Size);

    return Result;
}

tensor operator*(float scalar, const tensor &A)
{
    return A * scalar;
}

tensor operator/(const tensor &A, float scalar)
{
    tensor Result;
    Result.m_NDims = A.m_NDims;
    Result.m_Size = A.m_Size;
    for (int i = 0; i < A.m_NDims; i++)
    {
        Result.m_Dims[i] = A.m_Dims[i];
    }

    cudaMalloc(&Result.m_Data, Result.m_Size * sizeof(float));
    DivideScalarKernel<<<Ceil(Result.m_Size, 256), 256>>>(A.m_Data, Result.m_Data, scalar, Result.m_Size);

    return Result;
}

#endif // TENSOR_H
