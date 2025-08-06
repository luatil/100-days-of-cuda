#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#define F32_ABS(_X) fabs(_X)
#define ASSERT_CLOSE(_A, _B, _Tolerance)                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        if (F32_ABS((_A) - (_B)) >= (_Tolerance))                                                                      \
        {                                                                                                              \
            printf("ASSERTION FAILED: %f != %f (diff: %f, tolerance: %f) at %s:%d\n", (float)(_A), (float)(_B),        \
                   (float)F32_ABS((_A) - (_B)), (float)(_Tolerance), __FILE__, __LINE__);                              \
            asm("trap;");                                                                                              \
        }                                                                                                              \
    } while (0)

__global__ void VectorAddKernel(float *A, float *B, float *C, int N)
{
    int Tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (Tid < N)
    {
        C[Tid] = A[Tid] + B[Tid];
    }
}

__global__ void CompareFloatArrayKernel(float *A, float *B, float Tolerance, int N)
{
    int Tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (Tid < N)
    {
        ASSERT_CLOSE(A[Tid], B[Tid], Tolerance);
    }
}

extern "C" void CompareFloatArray(float *A, float *B, float Tolerance, int N)
{
    int BlockDim = 256;
    int GridDim = (N + BlockDim - 1) / BlockDim;
    CompareFloatArrayKernel<<<GridDim, BlockDim>>>(A, B, Tolerance, N);

    cudaError_t Err = cudaDeviceSynchronize();
    if (Err != cudaSuccess)
    {
        // printf("CUDA error during comparison: %s\n", cudaGetErrorString(Err));
        exit(1);
    }
}

extern "C" void VectorAdd(float *A, float *B, float *C, int N)
{
    int BlockDim = 256;
    int GridDim = (N + BlockDim - 1) / BlockDim;
    VectorAddKernel<<<GridDim, BlockDim>>>(A, B, C, N);
}

void TestVectorAdd(float *A, float *B, float *C, float *E, int N)
{
    float *DA, *DB, *DC, *DE;
    cudaMalloc(&DA, sizeof(float) * N);
    cudaMalloc(&DB, sizeof(float) * N);
    cudaMalloc(&DC, sizeof(float) * N);
    cudaMalloc(&DE, sizeof(float) * N);

    cudaMemcpy(DA, A, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(DB, B, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(DC, C, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(DE, E, sizeof(float) * N, cudaMemcpyHostToDevice);

    VectorAdd(DA, DB, DC, N);
    cudaDeviceSynchronize();

    CompareFloatArray(DC, DE, 0.001, N);
    cudaDeviceSynchronize();

    cudaFree(DA);
    cudaFree(DB);
    cudaFree(DC);
    cudaFree(DE);
}

void TestVectorAdd001()
{
    float A[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float B[] = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    float C[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float E[] = {6.0f, 6.0f, 6.0f, 6.0f, 6.0f};

    TestVectorAdd(A, B, C, E, 5);
}

int main()
{
    TestVectorAdd001();
}
