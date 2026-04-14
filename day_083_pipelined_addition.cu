#include <stdlib.h>

// Same kernels as day_083_ncu_study.cu — the difference is in how
// data is transferred and when the kernels are launched.
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
    const int N = 1 << 26; // same as ncu_study for a fair comparison
    const size_t Bytes = N * sizeof(float);

    // Split the work across N_STREAMS streams. Each stream owns a contiguous
    // chunk and pipelines its HtoD transfer → kernel → DtoH transfer.
    // While stream k is computing, stream k+1 can be transferring the next
    // chunk — overlapping communication with computation.
    const int N_STREAMS = 4;
    const int ChunkSize = N / N_STREAMS;
    const size_t ChunkBytes = ChunkSize * sizeof(float);

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

    cudaStream_t Streams[N_STREAMS];
    for (int I = 0; I < N_STREAMS; I++)
        cudaStreamCreate(&Streams[I]);

    const int NumOfThreads = 256;
    const int NumOfBlocks = (ChunkSize + NumOfThreads - 1) / NumOfThreads;

    for (int I = 0; I < N_STREAMS; I++)
    {
        int Offset = I * ChunkSize;
        cudaStream_t S = Streams[I];

        cudaMemcpyAsync(d_A + Offset, A + Offset, ChunkBytes, cudaMemcpyHostToDevice, S);
        cudaMemcpyAsync(d_B + Offset, B + Offset, ChunkBytes, cudaMemcpyHostToDevice, S);
        AddKernel<<<NumOfBlocks, NumOfThreads, 0, S>>>(d_A + Offset, d_B + Offset, d_C_add + Offset, ChunkSize);
        SubKernel<<<NumOfBlocks, NumOfThreads, 0, S>>>(d_A + Offset, d_B + Offset, d_C_sub + Offset, ChunkSize);
        cudaMemcpyAsync(C_add + Offset, d_C_add + Offset, ChunkBytes, cudaMemcpyDeviceToHost, S);
        cudaMemcpyAsync(C_sub + Offset, d_C_sub + Offset, ChunkBytes, cudaMemcpyDeviceToHost, S);
    }

    for (int I = 0; I < N_STREAMS; I++)
        cudaStreamSynchronize(Streams[I]);

    for (int I = 0; I < N_STREAMS; I++)
        cudaStreamDestroy(Streams[I]);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_add);
    cudaFree(d_C_sub);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C_add);
    cudaFreeHost(C_sub);
}
