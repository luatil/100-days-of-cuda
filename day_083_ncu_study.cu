#include <stdlib.h>

__global__ void AddKernel(float *A, float *B, float *C, int N)
{
    int Tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (Tid < N)
        C[Tid] = A[Tid] + B[Tid];
}

__global__ void SubKernel(float *A, float *B, float *C, int N)
{
    int Tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (Tid < N)
        C[Tid] = A[Tid] - B[Tid];
}

int main()
{
    const int N = 1 << 26; // 64M elements, ~256MB per array
    const size_t Bytes = N * sizeof(float);

    // Pinned host memory enables async transfers
    float *A, *B, *C_add, *C_sub;
    cudaMallocHost(&A, Bytes);
    cudaMallocHost(&B, Bytes);
    cudaMallocHost(&C_add, Bytes);
    cudaMallocHost(&C_sub, Bytes);

    for (int I = 0; I < N; I++)
    {
        A[I] = 3.0f;
        B[I] = 1.0f;
    }

    float *d_A, *d_B, *d_C_add, *d_C_sub;
    cudaMalloc(&d_A, Bytes);
    cudaMalloc(&d_B, Bytes);
    cudaMalloc(&d_C_add, Bytes);
    cudaMalloc(&d_C_sub, Bytes);

    // Transfer inputs once on the default stream
    cudaMemcpy(d_A, A, Bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, Bytes, cudaMemcpyHostToDevice);

    // Launch both kernels concurrently on separate streams.
    // nsys will show these overlapping in the timeline.
    cudaStream_t StreamAdd, StreamSub;
    cudaStreamCreate(&StreamAdd);
    cudaStreamCreate(&StreamSub);

    const int NumOfThreads = 256;
    const int NumOfBlocks = (N + NumOfThreads - 1) / NumOfThreads;

    AddKernel<<<NumOfBlocks, NumOfThreads, 0, StreamAdd>>>(d_A, d_B, d_C_add, N);
    SubKernel<<<NumOfBlocks, NumOfThreads, 0, StreamSub>>>(d_A, d_B, d_C_sub, N);

    cudaMemcpyAsync(C_add, d_C_add, Bytes, cudaMemcpyDeviceToHost, StreamAdd);
    cudaMemcpyAsync(C_sub, d_C_sub, Bytes, cudaMemcpyDeviceToHost, StreamSub);

    cudaStreamSynchronize(StreamAdd);
    cudaStreamSynchronize(StreamSub);

    cudaStreamDestroy(StreamAdd);
    cudaStreamDestroy(StreamSub);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_add);
    cudaFree(d_C_sub);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C_add);
    cudaFreeHost(C_sub);
}
