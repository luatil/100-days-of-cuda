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
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if (Row < M && Col < N)
    {
        float Sum = 0.0f;
        for (int K = 0; K < K; K++)
        {
            Sum += A[Row * K + K] * B[K * N + Col];
        }
        C[Row * N + Col] = Sum;
    }
}

__device__ __host__ int Ceil(int Num, int Den)
{
    return (Num + Den - 1) / Den;
}

struct tensor
{
    float *MData;
    int MDims[MAX_TENSOR_DIMS];
    int MNDims;
    size_t MSize;

    template <typename... Args> tensor(Args... Values)
    {
        static_assert(sizeof...(Args) > 0, "Must provide at least one value");

        MNDims = 1;
        MDims[0] = sizeof...(Args);
        MSize = sizeof...(Args);

        cudaMalloc(&MData, MSize * sizeof(float));

        float HostData[] = {static_cast<float>(Values)...};
        cudaMemcpy(MData, HostData, MSize * sizeof(float), cudaMemcpyHostToDevice);
    }

    tensor(tensor &&Other) noexcept : MData(Other.MData), MNDims(Other.MNDims), MSize(Other.MSize)
    {
        for (int I = 0; I < MNDims; I++)
        {
            MDims[I] = Other.MDims[I];
        }
        Other.MData = nullptr;
    }

    tensor &operator=(tensor &&Other) noexcept
    {
        if (this != &Other)
        {
            cudaFree(MData);
            MData = Other.MData;
            MNDims = Other.MNDims;
            MSize = Other.MSize;
            for (int I = 0; I < MNDims; I++)
            {
                MDims[I] = Other.MDims[I];
            }
            Other.MData = nullptr;
        }
        return *this;
    }

    tensor(const tensor &) = delete;
    tensor &operator=(const tensor &) = delete;

    ~tensor()
    {
        if (MData)
        {
            cudaFree(MData);
        }
    }

    size_t Size()
    {
        size_t Result = 1;
        for (int I = 0; I < MNDims; I++)
        {
            Result *= MDims[I];
        }
        return Result;
    }

    float *Data() const
    {
        return MData;
    }

    void Print() const
    {
        if (!MData || MSize == 0)
        {
            printf("[]\n");
            return;
        }

        float *HostData = (float *)malloc(MSize * sizeof(float));
        cudaMemcpy(HostData, MData, MSize * sizeof(float), cudaMemcpyDeviceToHost);

        if (MNDims == 1)
        {
            printf("[");
            for (size_t I = 0; I < MSize; I++)
            {
                printf("%.1f", HostData[I]);
                if (I < MSize - 1)
                    printf(", ");
            }
            printf("]\n");
        }
        else if (MNDims == 2)
        {
            printf("[");
            for (int I = 0; I < MDims[0]; I++)
            {
                printf("[");
                for (int J = 0; J < MDims[1]; J++)
                {
                    printf("%.1f", HostData[I * MDims[1] + J]);
                    if (J < MDims[1] - 1)
                        printf(", ");
                }
                printf("]");
                if (I < MDims[0] - 1)
                    printf(", ");
            }
            printf("]\n");
        }
        else
        {
            printf("Shape: (");
            for (int I = 0; I < MNDims; I++)
            {
                printf("%d", MDims[I]);
                if (I < MNDims - 1)
                    printf(", ");
            }
            printf(") Data: [");
            for (size_t I = 0; I < MSize; I++)
            {
                printf("%.1f", HostData[I]);
                if (I < MSize - 1)
                    printf(", ");
            }
            printf("]\n");
        }

        free(HostData);
    }

    template <typename... Args> tensor Reshape(Args... Dims)
    {
        static_assert(sizeof...(Args) > 0, "Must provide at least one dimension");

        int NewDims[] = {Dims...};
        int NewNdims = sizeof...(Args);

        size_t NewSize = 1;
        for (int I = 0; I < NewNdims; I++)
        {
            NewSize *= NewDims[I];
        }

        if (NewSize != MSize)
        {
            printf("Error: Cannot reshape tensor of size %zu to size %zu\n", MSize, NewSize);
            tensor Empty;
            return Empty;
        }

        tensor Result;
        Result.MData = MData;
        Result.MNDims = NewNdims;
        Result.MSize = MSize;

        for (int I = 0; I < NewNdims; I++)
        {
            Result.MDims[I] = NewDims[I];
        }

        MData = nullptr;

        return Result;
    }

    tensor MatMul(const tensor &B) const
    {
        if (MNDims != 2 || B.MNDims != 2)
        {
            printf("Error: MatMul requires both tensors to be 2D. Got %dD and %dD\n", MNDims, B.MNDims);
            tensor Empty;
            return Empty;
        }

        int M = MDims[0];   // Rows of A
        int K = MDims[1];   // Cols of A / Rows of B
        int N = B.MDims[1]; // Cols of B

        if (MDims[1] != B.MDims[0])
        {
            printf("Error: MatMul incompatible shapes: (%d, %d) × (%d, %d)\n", MDims[0], MDims[1], B.MDims[0],
                   B.MDims[1]);
            tensor Empty;
            return Empty;
        }

        tensor Result;
        Result.MNDims = 2;
        Result.MDims[0] = M;
        Result.MDims[1] = N;
        Result.MSize = M * N;

        cudaMalloc(&Result.MData, Result.MSize * sizeof(float));

        dim3 BlockSize(16, 16);
        dim3 GridSize((N + BlockSize.x - 1) / BlockSize.x, (M + BlockSize.y - 1) / BlockSize.y);

        MatMulKernel<<<GridSize, BlockSize>>>(MData, B.MData, Result.MData, M, N, K);

        return Result;
    }

    tensor &operator+=(const tensor &Other)
    {
        if (!IsShapeCompatible(Other))
        {
            printf("Error: Incompatible tensor shapes for += operation\n");
            return *this;
        }

        AddKernel<<<Ceil(MSize, 256), 256>>>(MData, Other.MData, MData, MSize);
        return *this;
    }

    tensor &operator-=(const tensor &Other)
    {
        if (!IsShapeCompatible(Other))
        {
            printf("Error: Incompatible tensor shapes for -= operation\n");
            return *this;
        }

        SubtractKernel<<<Ceil(MSize, 256), 256>>>(MData, Other.MData, MData, MSize);
        return *this;
    }

    tensor &operator*=(const tensor &Other)
    {
        if (!IsShapeCompatible(Other))
        {
            printf("Error: Incompatible tensor shapes for *= operation\n");
            return *this;
        }

        MultiplyKernel<<<Ceil(MSize, 256), 256>>>(MData, Other.MData, MData, MSize);
        return *this;
    }

    tensor &operator/=(const tensor &Other)
    {
        if (!IsShapeCompatible(Other))
        {
            printf("Error: Incompatible tensor shapes for /= operation\n");
            return *this;
        }

        DivideKernel<<<Ceil(MSize, 256), 256>>>(MData, Other.MData, MData, MSize);
        return *this;
    }

    tensor &operator+=(float Scalar)
    {
        AddScalarKernel<<<Ceil(MSize, 256), 256>>>(MData, MData, Scalar, MSize);
        return *this;
    }

    tensor &operator-=(float Scalar)
    {
        SubtractScalarKernel<<<Ceil(MSize, 256), 256>>>(MData, MData, Scalar, MSize);
        return *this;
    }

    tensor &operator*=(float Scalar)
    {
        MultiplyScalarKernel<<<Ceil(MSize, 256), 256>>>(MData, MData, Scalar, MSize);
        return *this;
    }

    tensor &operator/=(float Scalar)
    {
        DivideScalarKernel<<<Ceil(MSize, 256), 256>>>(MData, MData, Scalar, MSize);
        return *this;
    }

    tensor() : MData(nullptr), MNDims(0), MSize(0)
    {
    }

    bool IsShapeCompatible(const tensor &Other) const
    {
        if (MNDims != Other.MNDims || MSize != Other.MSize)
            return false;

        for (int I = 0; I < MNDims; I++)
        {
            if (MDims[I] != Other.MDims[I])
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
        tensor Empty;
        return Empty;
    }

    tensor Result;
    Result.MNDims = A.MNDims;
    Result.MSize = A.MSize;
    for (int I = 0; I < A.MNDims; I++)
    {
        Result.MDims[I] = A.MDims[I];
    }

    cudaMalloc(&Result.MData, Result.MSize * sizeof(float));
    AddKernel<<<Ceil(Result.MSize, 256), 256>>>(A.MData, B.MData, Result.MData, Result.MSize);

    return Result;
}

tensor operator-(const tensor &A, const tensor &B)
{
    if (!A.IsShapeCompatible(B))
    {
        printf("Error: Incompatible tensor shapes for subtraction\n");
        tensor Empty;
        return Empty;
    }

    tensor Result;
    Result.MNDims = A.MNDims;
    Result.MSize = A.MSize;
    for (int I = 0; I < A.MNDims; I++)
    {
        Result.MDims[I] = A.MDims[I];
    }

    cudaMalloc(&Result.MData, Result.MSize * sizeof(float));
    SubtractKernel<<<Ceil(Result.MSize, 256), 256>>>(A.MData, B.MData, Result.MData, Result.MSize);

    return Result;
}

tensor operator*(const tensor &A, const tensor &B)
{
    if (!A.IsShapeCompatible(B))
    {
        printf("Error: Incompatible tensor shapes for multiplication\n");
        tensor Empty;
        return Empty;
    }

    tensor Result;
    Result.MNDims = A.MNDims;
    Result.MSize = A.MSize;
    for (int I = 0; I < A.MNDims; I++)
    {
        Result.MDims[I] = A.MDims[I];
    }

    cudaMalloc(&Result.MData, Result.MSize * sizeof(float));
    MultiplyKernel<<<Ceil(Result.MSize, 256), 256>>>(A.MData, B.MData, Result.MData, Result.MSize);

    return Result;
}

tensor operator/(const tensor &A, const tensor &B)
{
    if (!A.IsShapeCompatible(B))
    {
        printf("Error: Incompatible tensor shapes for division\n");
        tensor Empty;
        return Empty;
    }

    tensor Result;
    Result.MNDims = A.MNDims;
    Result.MSize = A.MSize;
    for (int I = 0; I < A.MNDims; I++)
    {
        Result.MDims[I] = A.MDims[I];
    }

    cudaMalloc(&Result.MData, Result.MSize * sizeof(float));
    DivideKernel<<<Ceil(Result.MSize, 256), 256>>>(A.MData, B.MData, Result.MData, Result.MSize);

    return Result;
}

tensor operator+(const tensor &A, float Scalar)
{
    tensor Result;
    Result.MNDims = A.MNDims;
    Result.MSize = A.MSize;
    for (int I = 0; I < A.MNDims; I++)
    {
        Result.MDims[I] = A.MDims[I];
    }

    cudaMalloc(&Result.MData, Result.MSize * sizeof(float));
    AddScalarKernel<<<Ceil(Result.MSize, 256), 256>>>(A.MData, Result.MData, Scalar, Result.MSize);

    return Result;
}

tensor operator+(float Scalar, const tensor &A)
{
    return A + Scalar;
}

tensor operator-(const tensor &A, float Scalar)
{
    tensor Result;
    Result.MNDims = A.MNDims;
    Result.MSize = A.MSize;
    for (int I = 0; I < A.MNDims; I++)
    {
        Result.MDims[I] = A.MDims[I];
    }

    cudaMalloc(&Result.MData, Result.MSize * sizeof(float));
    SubtractScalarKernel<<<Ceil(Result.MSize, 256), 256>>>(A.MData, Result.MData, Scalar, Result.MSize);

    return Result;
}

tensor operator-(float Scalar, const tensor &A)
{
    tensor Result;
    Result.MNDims = A.MNDims;
    Result.MSize = A.MSize;
    for (int I = 0; I < A.MNDims; I++)
    {
        Result.MDims[I] = A.MDims[I];
    }

    cudaMalloc(&Result.MData, Result.MSize * sizeof(float));
    AddScalarKernel<<<Ceil(Result.MSize, 256), 256>>>(A.MData, Result.MData, -Scalar, Result.MSize);

    return Result;
}

tensor operator*(const tensor &A, float Scalar)
{
    tensor Result;
    Result.MNDims = A.MNDims;
    Result.MSize = A.MSize;
    for (int I = 0; I < A.MNDims; I++)
    {
        Result.MDims[I] = A.MDims[I];
    }

    cudaMalloc(&Result.MData, Result.MSize * sizeof(float));
    MultiplyScalarKernel<<<Ceil(Result.MSize, 256), 256>>>(A.MData, Result.MData, Scalar, Result.MSize);

    return Result;
}

tensor operator*(float Scalar, const tensor &A)
{
    return A * Scalar;
}

tensor operator/(const tensor &A, float Scalar)
{
    tensor Result;
    Result.MNDims = A.MNDims;
    Result.MSize = A.MSize;
    for (int I = 0; I < A.MNDims; I++)
    {
        Result.MDims[I] = A.MDims[I];
    }

    cudaMalloc(&Result.MData, Result.MSize * sizeof(float));
    DivideScalarKernel<<<Ceil(Result.MSize, 256), 256>>>(A.MData, Result.MData, Scalar, Result.MSize);

    return Result;
}

#endif // TENSOR_H
