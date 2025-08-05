#include "solve.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void GemmFP16(const half *A, const half *B, half *C, int M, int N, int K, float Alpha, float Beta)
{
    int Col = blockDim.x * blockIdx.x + threadIdx.x;
    int Row = blockDim.y * blockIdx.y + threadIdx.y;

    if (Col < N && Row < M)
    {
        float Res = 0.0f;
        for (int K = 0; K < K; K++)
        {
            Res += __half2float(A[Row * K + K]) * __half2float(B[Col + K * N]);
        }
        Res *= Alpha;
        Res += Beta * __half2float(C[Row * N + Col]);
        C[Row * N + Col] = __float2half(Res);
    }
}

// A, B, and C are device pointers
void Solve(const half *A, const half *B, half *C, int M, int N, int K, float Alpha, float Beta)
{
    dim3 BlockDim(32, 32);
    dim3 GridDim((M + 32 - 1) / 32, (N + 32 - 1) / 32);
    GemmFP16<<<GridDim, BlockDim>>>(A, B, C, M, N, K, Alpha, Beta);
}
